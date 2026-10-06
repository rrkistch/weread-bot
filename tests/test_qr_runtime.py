import asyncio
import secrets
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List
from unittest.mock import AsyncMock, patch

from aiohttp.test_utils import TestClient, TestServer

from tests.helpers import load_weread_bot


class RuntimeFixture(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.bot = load_weread_bot()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = self.bot.WeReadConfig()
        self.config.auth = self.bot.AuthConfig(
            "qr", str(Path(self.temp.name) / "auth.json")
        )
        self.config.web = self.bot.WebConfig(
            public_url="https://read.example.com",
            admin_password=secrets.token_urlsafe(24),
        )
        self.config.network.rate_limit = 0
        self.config.history.file = str(Path(self.temp.name) / "history.json")
        self.config.notification.enabled = False
        self.shutdown = asyncio.Event()
        self.summary = {}
        self.application = SimpleNamespace(
            config=self.config,
            is_shutdown_requested=self.shutdown.is_set,
            set_run_summary=lambda data: self.summary.update(data),
        )
        self.runtime = self.bot.QRRuntime(self.application)
        self.addAsyncCleanup(self.runtime.cancel_login)

    def bind_account(self, book: bool = True) -> None:
        self.runtime.store.update(
            {
                "account": {"vid": "12345", "name": "测试账号"},
                "cookies": {
                    "wr_vid": "12345",
                    "wr_skey": secrets.token_urlsafe(24),
                },
                "book": {"id": "12345", "title": "测试书"} if book else {},
            },
            0,
            bump=True,
        )

    def login_client(
        self, responses: List[Any], vid: str = "12345"
    ) -> SimpleNamespace:
        return SimpleNamespace(
            request=AsyncMock(),
            json=AsyncMock(side_effect=responses),
            verify_identity=AsyncMock(
                return_value={"vid": vid, "name": "测试账号"}
            ),
            renew=AsyncMock(),
            close=AsyncMock(),
            cookies={},
        )

    async def until(self, predicate: Callable[[], bool]) -> None:
        async def wait() -> None:
            while not predicate():
                await asyncio.sleep(0.005)

        await asyncio.wait_for(wait(), 2)


class QRLoginFlowTests(RuntimeFixture):
    async def test_login_phase_logs_do_not_include_credentials(self) -> None:
        uid = secrets.token_urlsafe(24)
        access = secrets.token_urlsafe(32)
        refresh = secrets.token_urlsafe(32)
        client = self.login_client(
            [
                {"uid": uid},
                {
                    "succeed": True,
                    "webLoginVid": "12345",
                    "accessToken": access,
                    "refreshToken": refresh,
                },
            ]
        )
        flow = self.bot.QRLoginAttempt(client, 0)
        self.runtime.flow = flow
        with self.assertLogs(level="INFO") as logs:
            await self.runtime._poll_login(flow)
        messages = "\n".join(logs.output)
        for value in (uid, access, refresh):
            self.assertNotIn(value, messages)
        for phase in (
            "prepare",
            "wait_confirmation",
            "verify_credentials",
            "success",
        ):
            self.assertIn("state=" + phase, messages)
        self.assertEqual(flow.status, "success")

    async def test_concurrent_cookie_snapshots_do_not_restore_old_values(
        self,
    ) -> None:
        self.bind_account()
        state = self.runtime.store.snapshot()
        first = self.runtime.client(state)
        second = self.runtime.client(state)
        self.addAsyncCleanup(first.close)
        self.addAsyncCleanup(second.close)
        renewed = secrets.token_urlsafe(24)
        refresh = secrets.token_urlsafe(24)
        first.save_cookies({**state["cookies"], "wr_skey": renewed})
        result = second.save_cookies({**state["cookies"], "wr_rt": refresh})
        self.assertEqual(result["wr_skey"], renewed)
        self.assertEqual(result["wr_rt"], refresh)
        self.assertEqual(self.runtime.store.state["cookies"], result)

    async def test_success_persists_credentials_and_resumes_existing_book(
        self,
    ):
        self.bind_account()
        access = secrets.token_urlsafe(32)
        refresh = secrets.token_urlsafe(32)
        client = self.login_client(
            [
                {"uid": secrets.token_urlsafe(20)},
                {
                    "succeed": True,
                    "webLoginVid": "12345",
                    "accessToken": access,
                    "refreshToken": refresh,
                },
            ]
        )
        flow = self.bot.QRLoginAttempt(client, 1)
        self.runtime.flow = flow
        await self.runtime._poll_login(flow)
        self.assertEqual(flow.status, "success")
        self.assertEqual(
            self.runtime.store.state["cookies"]["wr_skey"], access
        )
        self.assertEqual(self.runtime.store.state["revision"], 2)
        self.assertTrue(self.runtime.resume.is_set())
        client.verify_identity.assert_awaited_once()
        client.renew.assert_awaited_once()
        client.close.assert_awaited_once()

    async def test_first_login_waits_for_book(self) -> None:
        client = self.login_client(
            [
                {"uid": secrets.token_urlsafe(20)},
                {
                    "succeed": True,
                    "webLoginVid": "12345",
                    "accessToken": secrets.token_urlsafe(32),
                },
            ]
        )
        flow = self.bot.QRLoginAttempt(client, 0)
        self.runtime.flow = flow
        await self.runtime._poll_login(flow)
        self.assertEqual(flow.status, "success")
        self.assertFalse(self.runtime.ready())
        self.assertFalse(self.runtime.resume.is_set())

    async def test_otp_retry_then_success(self) -> None:
        client = self.login_client(
            [
                {"uid": secrets.token_urlsafe(20)},
                {"logicCode": "NEED_OTP"},
                {"logicCode": "OTP_NOT_MATCH"},
                {
                    "succeed": True,
                    "webLoginVid": "12345",
                    "accessToken": secrets.token_urlsafe(32),
                },
            ]
        )
        flow = self.bot.QRLoginAttempt(client, 0)
        self.runtime.flow = flow
        flow.task = asyncio.create_task(self.runtime._poll_login(flow))
        await self.until(lambda: flow.status == "otp")
        flow.otp = "1234"
        flow.otp_ready.set()
        await self.until(
            lambda: client.json.await_count == 3 and flow.status == "otp"
        )
        flow.otp = "5678"
        flow.otp_ready.set()
        await flow.task
        self.assertEqual(flow.status, "success")
        self.assertEqual(flow.otp, "")
        self.assertEqual(
            client.json.call_args_list[-1].kwargs["params"]["otp"], "5678"
        )

    async def test_expired_cancelled_and_unknown_flows_never_save(
        self,
    ) -> None:
        for result, expected in (
            ({"logicCode": "LOGIN_TIMEOUT"}, "expired"),
            ({"logicCode": "OTP_EXPIRED"}, "expired"),
            ({"unexpected": True}, "failed"),
        ):
            client = self.login_client(
                [{"uid": secrets.token_urlsafe(20)}, result]
            )
            flow = self.bot.QRLoginAttempt(client, 0)
            self.runtime.flow = flow
            await self.runtime._poll_login(flow)
            self.assertEqual(flow.status, expected)
            self.assertEqual(self.runtime.store.state["revision"], 0)
        await self.runtime.cancel_login()
        self.assertIsNone(self.runtime.flow)

    async def test_wrong_account_and_stale_qr_keep_old_credentials(
        self,
    ) -> None:
        self.bind_account()
        before = self.runtime.store.snapshot()
        client = self.login_client([])
        flow = self.bot.QRLoginAttempt(client, 1)
        self.runtime.flow = flow
        with self.assertRaisesRegex(self.bot.QRServiceError, "原账号"):
            await self.runtime._accept_login(
                flow, {"vid": "67890", "name": "另一个账号"}
            )
        self.assertEqual(self.runtime.store.snapshot(), before)
        self.runtime.flow = self.bot.QRLoginAttempt(client, 1)
        await self.runtime._accept_login(flow, before["account"])
        self.assertEqual(self.runtime.store.snapshot(), before)

    async def test_network_failure_is_retryable_and_preserves_account(
        self,
    ) -> None:
        self.bind_account()
        before = self.runtime.store.snapshot()
        client = self.login_client(
            [
                self.bot.QRServiceError(
                    "网络暂时不可用",
                    self.bot.RuntimeErrorCategory.NETWORK,
                )
            ]
        )
        flow = self.bot.QRLoginAttempt(client, 1)
        self.runtime.flow = flow
        await self.runtime._poll_login(flow)
        self.assertEqual(flow.status, "failed")
        self.assertEqual(self.runtime.store.snapshot(), before)

    async def test_relogin_stops_reading_before_committing_new_credentials(
        self,
    ):
        self.bind_account()
        old_revision = self.runtime.store.state["revision"]

        async def reading() -> None:
            await self.runtime.cancel_reading.wait()
            self.assertEqual(
                self.runtime.store.state["revision"], old_revision
            )

        self.runtime.active_task = asyncio.create_task(reading())
        client = self.login_client([])
        client.cookies = {"wr_skey": secrets.token_urlsafe(20)}
        flow = self.bot.QRLoginAttempt(client, old_revision)
        self.runtime.flow = flow
        await self.runtime._accept_login(
            flow, self.runtime.store.state["account"]
        )
        self.assertTrue(self.runtime.active_task.done())
        self.assertEqual(
            self.runtime.store.state["revision"], old_revision + 1
        )

    async def test_auth_notification_is_deduplicated_and_revision_guarded(
        self,
    ):
        self.bind_account()
        notify = AsyncMock()
        with patch.object(
            self.bot.NotificationService, "send_notification_async", notify
        ):
            await self.runtime.auth_failed(0)
            notify.assert_not_awaited()
            await self.runtime.auth_failed(1)
            await self.runtime.auth_failed(1)
        notify.assert_awaited_once()
        self.assertIn(self.config.web.public_url, notify.call_args.args[0])
        self.assertTrue(self.runtime.store.state["needs_login"])

    async def test_book_failure_keeps_previous_selection(self) -> None:
        self.bind_account()
        client = SimpleNamespace(
            shelf=AsyncMock(
                return_value=[{"id": "555", "title": "不可用的书"}]
            ),
            prepare_book=AsyncMock(
                side_effect=self.bot.QRServiceError("不可阅读")
            ),
            close=AsyncMock(),
        )
        before = self.runtime.store.snapshot()
        with patch.object(self.runtime, "client", return_value=client):
            with self.assertRaises(self.bot.QRServiceError):
                await self.runtime.select_book("555")
        self.assertEqual(self.runtime.store.snapshot(), before)

    async def test_resume_and_due_schedule_run_only_one_session(self) -> None:
        self.bind_account()
        self.config.startup_mode = "scheduled"
        self.config.schedule.enabled = True
        calls = []

        async def run_once(state: Dict[str, Any]) -> Any:
            calls.append(state["revision"])
            await asyncio.sleep(0)
            self.shutdown.set()
            return self.bot.SessionResult(
                self.bot.SessionStatus.SUCCESS, self.bot.ReadingSession()
            )

        self.runtime.resume.set()
        with (
            patch.object(self.bot.web, "AppRunner") as runner,
            patch.object(self.bot.web, "TCPSite") as site,
            patch.object(self.runtime, "_run_once", side_effect=run_once),
            patch.object(
                self.runtime,
                "_scheduled_time",
                return_value=datetime.now(self.runtime.tz)
                - timedelta(seconds=1),
            ),
        ):
            runner.return_value.setup = AsyncMock()
            runner.return_value.cleanup = AsyncMock()
            site.return_value.start = AsyncMock()
            await self.runtime.run()
        self.assertEqual(calls, [1])
        self.assertFalse(self.runtime.resume.is_set())

    async def test_daily_limit_survives_runtime_restart(self) -> None:
        self.bind_account()
        self.config.startup_mode = "daemon"
        self.config.daemon.max_daily_sessions = 2
        now = datetime.now(self.runtime.tz)
        self.runtime.store.update(
            {"daily_date": now.date().isoformat(), "daily_count": 2}, 1
        )
        restarted = self.bot.QRRuntime(self.application)
        self.assertTrue(restarted._daily_limit(now))
        self.assertFalse(restarted._daily_limit(now + timedelta(days=1)))


class QRWebTests(RuntimeFixture):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.web_client = TestClient(
            TestServer(self.bot.build_qr_web_app(self.runtime))
        )
        await self.web_client.start_server()
        self.addAsyncCleanup(self.web_client.close)
        self.headers = {"Origin": self.config.web.public_url}

    async def login(self) -> str:
        response = await self.web_client.post(
            "/api/session",
            headers=self.headers,
            json={"password": self.config.web.admin_password},
        )
        self.assertEqual(response.status, 200)
        body = await response.json()
        cookie = response.cookies["wr_admin"]
        self.assertTrue(cookie["secure"])
        self.assertTrue(cookie["httponly"])
        self.assertEqual(cookie["samesite"], "Strict")
        # 本地 HTTP 测试显式发送 Cookie；产品始终要求 HTTPS Secure Cookie。
        self.headers.update(
            {
                "Cookie": "wr_admin=" + cookie.value,
                "X-CSRF-Token": body["csrf"],
            }
        )
        return cookie.value

    async def test_page_is_self_contained_with_nonce(self) -> None:
        response = await self.web_client.get("/")
        html = await response.text()
        self.assertEqual(response.status, 200)
        self.assertNotIn("__NONCE__", html)
        self.assertIn("nonce-", response.headers["Content-Security-Policy"])
        self.assertIn("no-store", response.headers["Cache-Control"])
        self.assertNotIn("https://cdn", html)

    async def test_private_endpoints_require_login(self) -> None:
        for path in ("status", "books", "qr/image"):
            response = await self.web_client.get("/api/" + path)
            self.assertEqual(response.status, 401)

    async def test_expired_session_and_logout_revoke_access(self) -> None:
        sid = await self.login()
        self.runtime.sessions[sid]["expires"] = 0
        response = await self.web_client.get(
            "/api/status", headers=self.headers
        )
        self.assertEqual(response.status, 401)
        await self.login()
        response = await self.web_client.delete(
            "/api/session", headers=self.headers
        )
        self.assertEqual(response.status, 200)
        response = await self.web_client.get(
            "/api/status", headers=self.headers
        )
        self.assertEqual(response.status, 401)

    async def test_origin_and_csrf_protect_mutations(self) -> None:
        await self.login()
        for override in (
            {"Origin": "https://other.example.com"},
            {"X-CSRF-Token": ""},
        ):
            response = await self.web_client.post(
                "/api/qr", headers={**self.headers, **override}, json={}
            )
            self.assertEqual(response.status, 403)
        self.assertIsNone(self.runtime.flow)

    async def test_password_login_is_rate_limited(self) -> None:
        for _ in range(5):
            response = await self.web_client.post(
                "/api/session",
                headers=self.headers,
                json={"password": secrets.token_urlsafe(24)},
            )
            self.assertEqual(response.status, 401)
        response = await self.web_client.post(
            "/api/session",
            headers=self.headers,
            json={"password": self.config.web.admin_password},
        )
        self.assertEqual(response.status, 429)

    async def test_status_never_exposes_upstream_secrets(self) -> None:
        self.bind_account()
        await self.login()
        response = await self.web_client.get(
            "/api/status", headers=self.headers
        )
        text = await response.text()
        self.assertEqual(response.status, 200)
        self.assertNotIn(self.runtime.store.state["cookies"]["wr_skey"], text)
        self.assertNotIn("cookies", text)
        self.assertEqual((await response.json())["account_name"], "测试账号")

    async def test_qr_image_and_otp_are_bound_to_current_attempt(self) -> None:
        await self.login()
        client = self.login_client([])
        flow = self.bot.QRLoginAttempt(
            client, 0, uid=secrets.token_urlsafe(24), status="otp"
        )
        self.runtime.flow = flow
        response = await self.web_client.get(
            "/api/qr/image?id=" + flow.id, headers=self.headers
        )
        self.assertEqual(response.status, 200)
        self.assertEqual(response.content_type, "image/svg+xml")
        response = await self.web_client.post(
            "/api/qr/otp",
            headers=self.headers,
            json={"id": "stale", "otp": "1234"},
        )
        self.assertEqual(response.status, 409)
        response = await self.web_client.post(
            "/api/qr/otp",
            headers=self.headers,
            json={"id": flow.id, "otp": "1234"},
        )
        self.assertEqual(response.status, 200)
        self.assertTrue(flow.otp_ready.is_set())


class QRConfigTests(unittest.TestCase):
    def test_qr_diagnostics_need_no_curl_and_make_no_network_requests(
        self,
    ) -> None:
        bot = load_weread_bot()
        with tempfile.TemporaryDirectory() as directory:
            config = bot.WeReadConfig()
            config.auth = bot.AuthConfig(
                "qr", str(Path(directory) / "auth.json")
            )
            config.web = bot.WebConfig(
                public_url="https://read.example.com",
                admin_password=secrets.token_urlsafe(24),
            )
            config.logging.console = False
            config.logging.file = ""
            args = SimpleNamespace(
                config="missing",
                mode=None,
                verbose=False,
                dry_run=True,
                validate_config=False,
                show_last_run=False,
            )
            with (
                patch.object(bot, "parse_arguments", return_value=args),
                patch.object(
                    bot,
                    "ConfigManager",
                    return_value=SimpleNamespace(config=config),
                ),
                patch.object(bot, "setup_logging"),
                patch.object(bot, "QRRuntime") as runtime,
            ):
                self.assertEqual(asyncio.run(bot.main()), 0)
                runtime.assert_not_called()
            self.assertFalse(Path(config.auth.state_file).exists())

    def test_password_and_https_are_required(self) -> None:
        bot = load_weread_bot()
        config = bot.WeReadConfig(auth=bot.AuthConfig("qr"))
        with self.assertRaises(bot.ConfigError):
            bot._validate_runtime_config(config)
        config.web.admin_password = secrets.token_urlsafe(24)
        config.web.public_url = "http://read.example.com"
        with self.assertRaises(bot.ConfigError):
            bot._validate_runtime_config(config)
        config.web.public_url = "https://read.example.com:443/"
        bot._validate_runtime_config(config)
        self.assertEqual(config.web.public_url, "https://read.example.com")


if __name__ == "__main__":
    unittest.main()
