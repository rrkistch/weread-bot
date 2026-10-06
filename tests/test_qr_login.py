import asyncio
import hashlib
import json
import secrets
import tempfile
import unittest
from pathlib import Path
from typing import Callable
from unittest.mock import patch

import httpx

from tests.helpers import load_weread_bot


class AuthStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bot = load_weread_bot()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "auth.json"

    def test_atomic_store_restart_and_stale_writer(self) -> None:
        store = self.bot.AuthStateStore(self.path)
        first = secrets.token_urlsafe(32)
        second = secrets.token_urlsafe(32)
        self.assertTrue(
            store.update({"cookies": {"wr_skey": first}}, 0, bump=True)
        )
        self.assertFalse(store.update({"cookies": {"wr_skey": second}}, 0))
        restarted = self.bot.AuthStateStore(self.path)
        self.assertEqual(restarted.state["cookies"]["wr_skey"], first)
        self.assertEqual(
            restarted.state["device_fingerprint"],
            store.state["device_fingerprint"],
        )
        self.assertEqual(restarted.state["revision"], 1)

    def test_failed_replace_keeps_previous_disk_and_memory(self) -> None:
        store = self.bot.AuthStateStore(self.path)
        before = self.path.read_bytes()
        with patch.object(self.bot.os, "replace", side_effect=OSError()):
            with self.assertRaises(self.bot.QRServiceError):
                store.update({"needs_login": True}, 0)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertFalse(store.state["needs_login"])
        self.assertEqual(list(self.path.parent.glob(".auth-*.tmp")), [])

    def test_corrupt_state_is_not_overwritten(self) -> None:
        self.path.write_text("{broken", encoding="utf-8")
        with self.assertRaises(self.bot.ConfigError):
            self.bot.AuthStateStore(self.path)
        self.assertEqual(self.path.read_text(), "{broken")

    def test_merge_cookies_honors_expiry_deletion_and_domain(self) -> None:
        fresh = secrets.token_hex(16)
        response = httpx.Response(
            200,
            headers=[
                (
                    "set-cookie",
                    f"wr_skey={fresh}; Domain=.weread.qq.com; Path=/; HttpOnly",
                ),
                ("set-cookie", "wr_rt=; Max-Age=0; Path=/"),
                (
                    "set-cookie",
                    "wr_old=gone; Expires=Thu, 01 Jan 1970 00:00:00 GMT; Path=/",
                ),
                ("set-cookie", "foreign=ignored; Domain=example.com; Path=/"),
            ],
        )
        merged = self.bot.merge_response_cookies(
            {
                "wr_skey": secrets.token_hex(16),
                "wr_rt": secrets.token_hex(16),
                "wr_old": "expired",
                "other": "retained",
            },
            response,
        )
        self.assertEqual(merged, {"wr_skey": fresh, "other": "retained"})

    def test_reader_parser_never_evaluates_javascript(self) -> None:
        with self.assertRaises(self.bot.QRServiceError):
            self.bot.parse_reader_state(
                "window.__INITIAL_STATE__=function(){return {}}()"
            )
        state = {
            "reader": {
                "psvts": secrets.token_hex(8),
                "token": secrets.token_hex(8),
            }
        }
        self.assertEqual(
            self.bot.parse_reader_state(
                "window.__INITIAL_STATE__=" + json.dumps(state) + ";evil()"
            ),
            state["reader"],
        )
        state["reader"]["isBookForbidden"] = True
        with self.assertRaisesRegex(self.bot.QRServiceError, "不可阅读"):
            self.bot.parse_reader_state(
                "window.__INITIAL_STATE__=" + json.dumps(state)
            )

    def test_redaction_hides_login_query_and_structured_secrets(self) -> None:
        secret = secrets.token_urlsafe(20)
        text = self.bot.redact_for_log(
            "GET https://weread.qq.com/api/auth/getLoginInfo?uid="
            + secret
            + "&otp=1234"
        )
        self.assertNotIn(secret, text)
        self.assertNotIn("1234", text)
        for key in (
            "accessToken",
            "refreshToken",
            "otp",
            "uid",
            "wr_rt",
            "password",
        ):
            self.assertNotIn(
                secret, str(self.bot.redact_for_log({key: secret}))
            )


class QRClientTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.bot = load_weread_bot()
        self.client = self.bot.WeReadQRClient(
            self.bot.NetworkConfig(rate_limit=0),
            {
                "cookies": {
                    "wr_vid": "12345",
                    "wr_skey": secrets.token_urlsafe(32),
                },
                "device_fingerprint": "123456",
                "user_agent": self.bot.QR_USER_AGENT,
            },
        )
        self.addAsyncCleanup(self.client.close)

    async def transport(
        self, handler: Callable[[httpx.Request], httpx.Response]
    ) -> None:
        await self.client.http._client.aclose()
        self.client.http._client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        )

    async def test_login_handshake_does_not_wait_for_reading_limiter(
        self,
    ) -> None:
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request.url.path)
            return httpx.Response(200, json={"succ": 1})

        await self.transport(handler)
        gate = asyncio.Event()
        self.client.http._rate_limiter.acquire = gate.wait
        await asyncio.wait_for(
            self.client.json("GET", "/api/auth/getLoginUid"),
            0.5,
        )
        await asyncio.wait_for(
            self.client.json(
                "GET",
                "/api/auth/getLoginInfo",
                params={"uid": secrets.token_urlsafe(24), "otp": ""},
            ),
            0.5,
        )
        reading = asyncio.create_task(
            self.client.json("POST", "/web/book/read", {})
        )
        try:
            await asyncio.sleep(0)
            self.assertFalse(reading.done())
            self.assertNotIn("/web/book/read", requests)
        finally:
            gate.set()
            await reading
        self.assertEqual(
            requests,
            [
                "/api/auth/getLoginUid",
                "/api/auth/getLoginInfo",
                "/web/book/read",
            ],
        )

    async def test_empty_otp_keeps_equals_sign(self) -> None:
        queries = []

        def handler(request: httpx.Request) -> httpx.Response:
            queries.append(request.url.query)
            return httpx.Response(200, json={"logicCode": "LOGIN_TIMEOUT"})

        await self.transport(handler)
        await self.client.json(
            "GET",
            "/api/auth/getLoginInfo",
            params={"uid": secrets.token_urlsafe(24), "otp": ""},
        )
        self.assertTrue(queries[0].endswith(b"&otp="))

    async def test_request_updates_all_cookies_without_duplicates(
        self,
    ) -> None:
        value = secrets.token_urlsafe(20)
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200,
                json={"succ": 1},
                headers=[
                    ("set-cookie", f"wr_rt={value}; Path=/"),
                    ("set-cookie", f"wr_skey={value}; Path=/"),
                ],
            )

        await self.transport(handler)
        saved = []
        self.client.save_cookies = lambda data: saved.append(data.copy())
        await self.client.json("GET", "/api/userInfo")
        await self.client.json("GET", "/api/userInfo")
        self.assertEqual(saved[-1]["wr_rt"], value)
        self.assertEqual(requests[-1].headers["cookie"].count("wr_skey="), 1)
        self.assertEqual(len(self.client.http._client.cookies), 0)

    async def test_renew_network_failure_is_not_auth_failure(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("untrusted detail", request=request)

        await self.transport(handler)
        with self.assertRaises(self.bot.QRServiceError) as caught:
            await self.client.renew(False)
        self.assertEqual(
            caught.exception.category, self.bot.RuntimeErrorCategory.NETWORK
        )
        self.assertNotIn("untrusted", str(caught.exception))

    async def test_renew_explicit_expiry_pauses_auth(self) -> None:
        await self.transport(
            lambda request: httpx.Response(200, json={"errCode": -2012})
        )
        with self.assertRaises(self.bot.QRServiceError) as caught:
            await self.client.renew(False)
        self.assertEqual(
            caught.exception.category, self.bot.RuntimeErrorCategory.AUTH
        )

    async def test_unknown_renewal_response_is_protocol_error(self) -> None:
        await self.transport(
            lambda request: httpx.Response(200, json={"unrecognized": True})
        )
        with self.assertRaises(self.bot.QRServiceError) as caught:
            await self.client.renew(False)
        self.assertEqual(
            caught.exception.category, self.bot.RuntimeErrorCategory.PROTOCOL
        )

    async def test_invalid_json_and_http_errors_are_safe(self) -> None:
        for response in (
            httpx.Response(403, text="upstream"),
            httpx.Response(200, text="<html>upstream</html>"),
        ):
            await self.transport(lambda request: response)
            with self.assertRaises(self.bot.QRServiceError) as caught:
                await self.client.json("GET", "/")
            self.assertNotIn("upstream", str(caught.exception))

    async def test_empty_shelf_is_valid(self) -> None:
        await self.transport(
            lambda request: httpx.Response(200, json={"books": []})
        )
        self.assertEqual(await self.client.shelf(), [])

    async def test_prepare_book_uses_progress_and_falls_back_when_chapter_removed(
        self,
    ):
        book = {"id": "12345", "title": "测试书"}
        chapter_uid = 2
        catalog = {
            "data": [
                {
                    "bookId": "12345",
                    "updated": [
                        {
                            "chapterUid": 0,
                            "chapterIdx": 0,
                            "title": "目录",
                            "wordCount": 0,
                        },
                        {
                            "chapterUid": 1,
                            "chapterIdx": 4,
                            "title": "第一章",
                            "wordCount": 120,
                        },
                        {
                            "chapterUid": 2,
                            "chapterIdx": 9,
                            "title": "第二章",
                            "wordCount": 130,
                        },
                    ],
                }
            ]
        }
        state = {
            "reader": {
                "psvts": secrets.token_hex(16),
                "token": secrets.token_hex(16),
                "bookInfo": {"bookId": "12345"},
            }
        }

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("chapterInfos"):
                return httpx.Response(200, json=catalog)
            if request.url.path.endswith("getProgress"):
                return httpx.Response(
                    200, json={"book": {"chapterUid": chapter_uid}}
                )
            return httpx.Response(
                200, text="window.__INITIAL_STATE__=" + json.dumps(state)
            )

        await self.transport(handler)
        result = await self.client.prepare_book(book)
        self.assertEqual(result["chapter"]["uid"], 2)
        self.assertEqual(result["chapter"]["index"], 9)
        chapter_uid = 999
        result = await self.client.prepare_book(book)
        self.assertEqual(result["chapter"]["uid"], 1)

    async def test_dynamic_reader_token_and_ids_are_used_in_read_request(
        self,
    ) -> None:
        token = secrets.token_hex(16)
        context = {"psvts": secrets.token_hex(16), "token": token, "pclts": 0}
        chapter = {"uid": 1, "id": self.bot.encode_weread_id(1), "index": 7}
        prepared = {
            "book": {"id": "12345", "title": "测试书"},
            "chapters": [chapter],
            "chapter": chapter,
            "reader": context,
            "progress": {},
        }
        config = self.bot.WeReadConfig()
        config.network.rate_limit = 0
        session = self.bot.QRReadingSession(
            config, self.client, prepared, lambda: False
        )
        sent = []

        def handler(request: httpx.Request) -> httpx.Response:
            sent.append(json.loads(request.content))
            return httpx.Response(
                200, json={"succ": 1, "synckey": secrets.token_hex(8)}
            )

        await self.transport(handler)
        success, _ = await session._simulate_reading_request(1)
        self.assertTrue(success)
        payload = sent[0]
        expected = hashlib.sha256(
            f"{payload['ts']}{payload['rn']}{token}".encode()
        ).hexdigest()
        self.assertEqual(payload["sg"], expected)
        self.assertEqual(
            payload["pc"], self.bot.encode_weread_id(payload["ct"])
        )
        self.assertEqual(payload["ci"], 7)
        self.assertEqual(payload["ps"], context["psvts"])
        self.assertEqual(
            payload["appId"], self.bot.web_app_id(self.client.user_agent)
        )


if __name__ == "__main__":
    unittest.main()
