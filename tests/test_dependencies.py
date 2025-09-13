"""Tests for the dependencies module, particularly HybridBackgroundTasks."""

from unittest.mock import Mock, patch

import pytest
from fastapi import BackgroundTasks

from agent_memory_server.config import settings
from agent_memory_server.dependencies import (
    DocketBackgroundTasks,
    HybridBackgroundTasks,
    get_background_tasks,
)


class TestHybridBackgroundTasks:
    """Test the HybridBackgroundTasks class."""

    def setup_method(self):
        """Store original use_docket setting."""
        self.original_use_docket = settings.use_docket

    def teardown_method(self):
        """Restore original use_docket setting."""
        settings.use_docket = self.original_use_docket

    @pytest.mark.asyncio
    async def test_add_task_with_fastapi_background_tasks(self):
        """Test that tasks are added to FastAPI background tasks when use_docket=False."""
        settings.use_docket = False

        bg_tasks = HybridBackgroundTasks()

        def test_func(arg1, arg2=None):
            return f"test_func called with {arg1}, {arg2}"

        # Add task
        await bg_tasks.add_task(test_func, "hello", arg2="world")

        # Verify task was added to FastAPI background tasks
        assert len(bg_tasks.tasks) == 1
        task = bg_tasks.tasks[0]
        assert task.func is test_func
        assert task.args == ("hello",)
        assert task.kwargs == {"arg2": "world"}

    @pytest.mark.asyncio
    async def test_add_task_with_docket(self):
        """Test that tasks are scheduled through Docket when use_docket=True."""
        settings.use_docket = True

        with patch("docket.Docket") as mock_docket_class:
            # Mock the docket instance and context manager
            mock_docket_instance = Mock()
            mock_docket_class.return_value.__aenter__.return_value = (
                mock_docket_instance
            )

            # Mock the add method - add returns a callable that returns an awaitable
            def mock_task_callable(*args, **kwargs):
                async def execution():
                    return f"Task executed with args={args}, kwargs={kwargs}"

                return execution()

            # docket.add(func) returns a callable
            mock_docket_instance.add.return_value = mock_task_callable

            bg_tasks = HybridBackgroundTasks()

            async def test_func(arg1, arg2=None):
                return f"test_func called with {arg1}, {arg2}"

            # Add task
            await bg_tasks.add_task(test_func, "hello", arg2="world")

            # Verify Docket was used
            mock_docket_class.assert_called_once_with(
                name=settings.docket_name,
                url=settings.redis_url,
            )
            mock_docket_instance.add.assert_called_once_with(test_func)

    @pytest.mark.asyncio
    async def test_add_task_logs_correctly_for_fastapi(self):
        """Test that FastAPI background tasks work without errors."""
        settings.use_docket = False

        bg_tasks = HybridBackgroundTasks()

        def test_func():
            pass

        # Should not raise any errors
        await bg_tasks.add_task(test_func)

        # Verify task was added to FastAPI background tasks
        assert len(bg_tasks.tasks) == 1

    @pytest.mark.asyncio
    async def test_add_task_logs_correctly_for_docket(self):
        """Test that Docket tasks work without errors."""
        settings.use_docket = True

        with patch("docket.Docket") as mock_docket_class:
            mock_docket_instance = Mock()
            mock_docket_class.return_value.__aenter__.return_value = (
                mock_docket_instance
            )

            def mock_task_callable(*args, **kwargs):
                async def execution():
                    return "executed"

                return execution()

            mock_docket_instance.add.return_value = mock_task_callable

            bg_tasks = HybridBackgroundTasks()

            def test_func():
                pass

            # Should not raise any errors
            await bg_tasks.add_task(test_func)

            # Verify Docket was called
            mock_docket_instance.add.assert_called_once_with(test_func)

    def test_inherits_from_background_tasks(self):
        """Test that HybridBackgroundTasks inherits from FastAPI BackgroundTasks."""
        bg_tasks = HybridBackgroundTasks()
        assert isinstance(bg_tasks, BackgroundTasks)

    def test_backward_compatibility_alias(self):
        """Test that DocketBackgroundTasks is an alias for HybridBackgroundTasks."""
        assert DocketBackgroundTasks is HybridBackgroundTasks


class TestGetBackgroundTasks:
    """Test the get_background_tasks dependency function."""

    def test_returns_hybrid_background_tasks_instance(self):
        """Test that get_background_tasks returns a HybridBackgroundTasks instance."""
        bg_tasks = get_background_tasks()

        assert isinstance(bg_tasks, HybridBackgroundTasks)
        assert isinstance(bg_tasks, BackgroundTasks)

    def test_returns_new_instance_each_call(self):
        """Test that get_background_tasks returns a new instance each time."""
        bg_tasks1 = get_background_tasks()
        bg_tasks2 = get_background_tasks()

        assert bg_tasks1 is not bg_tasks2
        assert isinstance(bg_tasks1, HybridBackgroundTasks)
        assert isinstance(bg_tasks2, HybridBackgroundTasks)


class TestIntegrationWithSettings:
    """Test integration with the settings configuration."""

    def setup_method(self):
        """Store original use_docket setting."""
        self.original_use_docket = settings.use_docket

    def teardown_method(self):
        """Restore original use_docket setting."""
        settings.use_docket = self.original_use_docket

    @pytest.mark.asyncio
    async def test_respects_settings_change(self):
        """Test that HybridBackgroundTasks respects runtime changes to settings.use_docket."""
        bg_tasks = HybridBackgroundTasks()

        def test_func():
            pass

        # Test with use_docket=False
        settings.use_docket = False
        await bg_tasks.add_task(test_func)
        assert len(bg_tasks.tasks) == 1

        # Clear tasks and test with use_docket=True
        bg_tasks.tasks.clear()
        settings.use_docket = True

        with patch("docket.Docket") as mock_docket_class:
            mock_docket_instance = Mock()
            mock_docket_class.return_value.__aenter__.return_value = (
                mock_docket_instance
            )

            def mock_task_callable(*args, **kwargs):
                async def execution():
                    return "executed"

                return execution()

            mock_docket_instance.add.return_value = mock_task_callable

            await bg_tasks.add_task(test_func)

            # Should not add to FastAPI tasks when use_docket=True
            assert len(bg_tasks.tasks) == 0
            # Should have called Docket
            mock_docket_class.assert_called_once()

    def test_uses_correct_docket_settings(self):
        """Test that HybridBackgroundTasks uses the correct docket name and URL from settings."""
        settings.use_docket = True

        with patch("docket.Docket") as mock_docket_class:
            mock_docket_instance = Mock()
            mock_docket_class.return_value.__aenter__.return_value = (
                mock_docket_instance
            )

            def mock_task_callable(*args, **kwargs):
                async def execution():
                    return "executed"

                return execution()

            mock_docket_instance.add.return_value = mock_task_callable

            bg_tasks = HybridBackgroundTasks()

            async def run_test():
                await bg_tasks.add_task(lambda: None)

            import asyncio

            asyncio.run(run_test())

            # Verify Docket was initialized with correct settings
            mock_docket_class.assert_called_once_with(
                name=settings.docket_name,
                url=settings.redis_url,
            )
