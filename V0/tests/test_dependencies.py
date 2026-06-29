"""Tests for the dependencies module, particularly HybridBackgroundTasks."""

import asyncio
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
    async def test_add_task_executes_async_function(self):
        """Test that async tasks are executed when use_docket=False."""
        settings.use_docket = False

        bg_tasks = HybridBackgroundTasks()
        result = {"called": False, "args": None}

        async def test_func(arg1, arg2=None):
            result["called"] = True
            result["args"] = (arg1, arg2)

        # Add task
        bg_tasks.add_task(test_func, "hello", arg2="world")

        # Give the task time to execute
        await asyncio.sleep(0.1)

        # Verify task was executed with correct arguments
        assert result["called"] is True
        assert result["args"] == ("hello", "world")

    @pytest.mark.asyncio
    async def test_add_task_executes_sync_function(self):
        """Test that sync tasks are executed in a thread pool when use_docket=False."""
        settings.use_docket = False

        bg_tasks = HybridBackgroundTasks()
        result = {"called": False, "args": None}

        def test_func(arg1, arg2=None):
            result["called"] = True
            result["args"] = (arg1, arg2)

        # Add task
        bg_tasks.add_task(test_func, "hello", arg2="world")

        # Give the task time to execute
        await asyncio.sleep(0.1)

        # Verify task was executed with correct arguments
        assert result["called"] is True
        assert result["args"] == ("hello", "world")

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

            # Add task - this should schedule directly in Docket
            bg_tasks.add_task(test_func, "hello", arg2="world")

            # Give the async task a moment to complete
            await asyncio.sleep(0.1)

            # When use_docket=True, tasks should NOT be added to FastAPI background tasks
            assert len(bg_tasks.tasks) == 0

            # Verify Docket was used
            mock_docket_class.assert_called_once_with(
                name=settings.docket_name,
                url=settings.redis_url,
            )
            mock_docket_instance.add.assert_called_once_with(test_func)

    @pytest.mark.asyncio
    async def test_add_task_handles_errors_gracefully(self):
        """Test that task errors are logged but don't crash the system."""
        settings.use_docket = False

        bg_tasks = HybridBackgroundTasks()

        async def failing_task():
            raise ValueError("Test error")

        # Should not raise any errors when adding the task
        bg_tasks.add_task(failing_task)

        # Give the task time to execute (and fail)
        await asyncio.sleep(0.1)

        # If we got here, the error was handled gracefully

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
            bg_tasks.add_task(test_func)

            # Give the async task a moment to complete
            await asyncio.sleep(0.1)

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
        result = {"asyncio_called": False, "docket_called": False}

        async def test_func():
            result["asyncio_called"] = True

        # Test with use_docket=False - should use asyncio.create_task
        settings.use_docket = False
        bg_tasks = HybridBackgroundTasks()
        bg_tasks.add_task(test_func)

        # Give the task time to execute
        await asyncio.sleep(0.1)

        assert result["asyncio_called"] is True

        # Test with use_docket=True - should use Docket
        settings.use_docket = True

        with patch("docket.Docket") as mock_docket_class:
            mock_docket_instance = Mock()
            mock_docket_class.return_value.__aenter__.return_value = (
                mock_docket_instance
            )

            def mock_task_callable(*args, **kwargs):
                async def execution():
                    result["docket_called"] = True
                    return "executed"

                return execution()

            mock_docket_instance.add.return_value = mock_task_callable

            bg_tasks2 = HybridBackgroundTasks()
            bg_tasks2.add_task(test_func)

            # Give the async task a moment to complete
            await asyncio.sleep(0.1)

            # Should have called Docket
            mock_docket_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_uses_correct_docket_settings(self):
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

            bg_tasks.add_task(lambda: None)

            # Give the async task a moment to complete
            await asyncio.sleep(0.1)

            # Verify Docket was initialized with correct settings
            mock_docket_class.assert_called_once_with(
                name=settings.docket_name,
                url=settings.redis_url,
            )
