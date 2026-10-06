from datetime import datetime, tzinfo
from typing import Any, Callable, Dict, Optional
from unittest.mock import AsyncMock, patch

from tests.test_qr_runtime import RuntimeFixture


class DailyDaemonScheduleTests(RuntimeFixture):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.config.startup_mode = "daemon"
        self.config.daemon.enabled = True
        self.config.daemon.daily_start_time = "05:00"
        self.config.daemon.max_daily_sessions = 3
        self.config.daemon.session_interval = "120-180"
        self.bind_account()

    def at(self, hour: int, minute: int = 0, day: int = 6) -> datetime:
        return datetime(2026, 10, day, hour, minute, tzinfo=self.runtime.tz)

    def count(self, number: int, day: int = 6) -> None:
        self.runtime.store.update(
            {
                "daily_date": self.at(5, day=day).date().isoformat(),
                "daily_count": number,
            },
            self.runtime.store.state["revision"],
        )

    async def test_first_run_waits_until_five_and_catches_up_after_five(
        self,
    ) -> None:
        self.assertEqual(
            self.runtime._restore_daemon_next_run(self.at(4, 59)), self.at(5)
        )
        self.assertEqual(
            self.runtime._restore_daemon_next_run(self.at(5)), self.at(5)
        )
        self.assertEqual(
            self.runtime._restore_daemon_next_run(self.at(17)), self.at(17)
        )

    async def test_full_day_waits_for_next_five_not_midnight(self) -> None:
        self.count(3)
        self.assertEqual(
            self.runtime._next_daemon_session(self.at(15)), self.at(5, day=7)
        )
        self.assertEqual(
            self.runtime._restore_daemon_next_run(self.at(22)),
            self.at(5, day=7),
        )

    async def test_interval_is_measured_from_completion_and_survives_restart(
        self,
    ) -> None:
        self.count(1)
        with patch.object(
            self.bot.RandomHelper, "get_random_from_range", return_value=150
        ) as draw:
            next_run = self.runtime._next_daemon_session(self.at(6, 20))
        draw.assert_called_once_with("120-180")
        self.assertEqual(next_run, self.at(8, 50))
        self.runtime._save_daemon_next_run(next_run)
        restarted = self.bot.QRRuntime(self.application)
        with patch.object(
            self.bot.RandomHelper,
            "get_random_from_range",
            side_effect=AssertionError("must not redraw"),
        ):
            self.assertEqual(
                restarted._restore_daemon_next_run(self.at(7)), next_run
            )
            self.assertEqual(
                restarted._restore_daemon_next_run(self.at(9)), self.at(9)
            )

    async def test_crash_after_counting_waits_instead_of_repeating_session(
        self,
    ) -> None:
        self.count(1)
        self.runtime._save_daemon_next_run(None)
        restarted = self.bot.QRRuntime(self.application)
        with patch.object(
            self.bot.RandomHelper, "get_random_from_range", return_value=120
        ):
            self.assertEqual(
                restarted._restore_daemon_next_run(self.at(6)), self.at(8)
            )

    async def test_crossing_midnight_defers_to_next_daily_start(self) -> None:
        self.count(2)
        with patch.object(
            self.bot.RandomHelper, "get_random_from_range", return_value=180
        ):
            self.assertEqual(
                self.runtime._next_daemon_session(self.at(23)),
                self.at(5, day=7),
            )
        self.assertEqual(
            self.runtime._restore_daemon_next_run(self.at(4, day=7)),
            self.at(5, day=7),
        )
        self.assertEqual(self.runtime.store.state["daily_count"], 0)

    async def test_future_plan_with_changed_settings_is_not_reused(
        self,
    ) -> None:
        self.count(1)
        self.runtime._save_daemon_next_run(self.at(10))
        self.config.daemon.session_interval = "60-90"
        restarted = self.bot.QRRuntime(self.application)
        with patch.object(
            self.bot.RandomHelper, "get_random_from_range", return_value=60
        ):
            self.assertEqual(
                restarted._restore_daemon_next_run(self.at(7)), self.at(8)
            )

    async def test_login_resume_cannot_start_before_five(self) -> None:
        now = self.at(4)

        class Clock(datetime):
            @classmethod
            def now(cls, tz: Optional[tzinfo] = None) -> datetime:
                return now

        async def stop_sleep(*args: Any, **kwargs: Any) -> bool:
            self.shutdown.set()
            return False

        self.runtime.resume.set()
        with (
            patch.object(self.bot, "datetime", Clock),
            patch.object(
                self.bot, "interruptible_sleep", side_effect=stop_sleep
            ),
            patch.object(self.bot.web, "AppRunner") as runner,
            patch.object(self.bot.web, "TCPSite") as site,
            patch.object(
                self.runtime, "_run_once", new_callable=AsyncMock
            ) as read,
        ):
            runner.return_value.setup = AsyncMock()
            runner.return_value.cleanup = AsyncMock()
            site.return_value.start = AsyncMock()
            await self.runtime.run()
        read.assert_not_awaited()
        self.assertEqual(self.runtime.next_run, self.at(5))
        self.assertFalse(self.runtime.resume.is_set())

    async def test_third_session_saves_next_day_schedule(self) -> None:
        self.count(2)
        now = self.at(12)

        class Clock(datetime):
            @classmethod
            def now(cls, tz: Optional[tzinfo] = None) -> datetime:
                return now

        async def read_once(state: Dict[str, Any]) -> Any:
            self.shutdown.set()
            return self.bot.SessionResult(
                self.bot.SessionStatus.SUCCESS, self.bot.ReadingSession()
            )

        self.runtime.resume.set()
        with (
            patch.object(self.bot, "datetime", Clock),
            patch.object(self.bot.web, "AppRunner") as runner,
            patch.object(self.bot.web, "TCPSite") as site,
            patch.object(self.runtime, "_run_once", side_effect=read_once),
        ):
            runner.return_value.setup = AsyncMock()
            runner.return_value.cleanup = AsyncMock()
            site.return_value.start = AsyncMock()
            await self.runtime.run()
        self.assertEqual(self.runtime.store.state["daily_count"], 3)
        self.assertEqual(self.runtime.next_run, self.at(5, day=7))
        self.assertEqual(
            self.runtime.store.state["daemon_next_run"],
            self.at(5, day=7).isoformat(),
        )

    async def test_legacy_daemon_also_obeys_first_start_time(self) -> None:
        now = self.at(4)

        class Clock(datetime):
            @classmethod
            def now(cls, tz: Optional[tzinfo] = None) -> datetime:
                return now

        async def stop_sleep(
            seconds: float, is_cancelled: Callable[[], bool]
        ) -> bool:
            self.assertEqual(seconds, 3600)
            self.shutdown.set()
            return False

        app = object.__new__(self.bot.WeReadApplication)
        app.config = self.config
        app.is_shutdown_requested = self.shutdown.is_set
        app.run_single_session = AsyncMock()
        with (
            patch.object(self.bot, "datetime", Clock),
            patch.object(
                self.bot, "interruptible_sleep", side_effect=stop_sleep
            ),
        ):
            await app._run_daemon_mode()
        app.run_single_session.assert_not_awaited()

    async def test_time_validation_and_old_default(self) -> None:
        for value in ("5:00", "25:00", "05:60", None, True):
            self.config.daemon.daily_start_time = value
            with self.assertRaisesRegex(
                self.bot.ConfigError, "daemon.daily_start_time"
            ):
                self.bot.validate_config_semantics(self.config)
        self.config.daemon.daily_start_time = ""
        self.bot.validate_config_semantics(self.config)
        self.assertEqual(
            self.bot.daemon_interval_start(self.at(23), 180, ""),
            self.at(2, day=7),
        )
