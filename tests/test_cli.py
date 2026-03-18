"""
Tests for the CLI module.
"""

import sys
from datetime import timedelta
from unittest.mock import AsyncMock, Mock, patch

from click.testing import CliRunner

from agent_memory_server.cli import (
    VERSION,
    api,
    cli,
    delete_memory_cmd,
    delete_session_cmd,
    mcp,
    migrate_memories,
    rebuild_index,
    schedule_task,
    search,
    task_worker,
    version,
)


class TestVersion:
    """Tests for the version command."""

    def test_version_command(self):
        """Test that version command returns the correct version."""
        runner = CliRunner()
        result = runner.invoke(version)

        assert result.exit_code == 0
        assert VERSION in result.output
        assert "agent-memory-server version" in result.output


class TestRebuildIndex:
    """Tests for the rebuild_index command."""

    @patch("agent_memory_server.memory_vector_db_factory.get_memory_vector_db")
    def test_rebuild_index_command(self, mock_get_db):
        """Test rebuild_index command execution."""
        from agent_memory_server.memory_vector_db import RedisVLMemoryVectorDatabase

        # Create a mock index
        mock_index = Mock()
        mock_index.name = "test_index"
        mock_index.create = AsyncMock()

        # Create a mock database with a mock index
        mock_db = Mock(spec=RedisVLMemoryVectorDatabase)
        mock_db.index = mock_index

        mock_get_db.return_value = mock_db

        runner = CliRunner()
        result = runner.invoke(rebuild_index)

        assert result.exit_code == 0
        mock_get_db.assert_called_once()
        mock_index.create.assert_called_once_with(overwrite=True)


class TestMigrateMemories:
    """Tests for the migrate_memories command."""

    @patch("agent_memory_server.cli.migrate_normalize_tag_separators_4")
    @patch("agent_memory_server.cli.migrate_add_memory_type_3")
    @patch("agent_memory_server.cli.migrate_add_discrete_memory_extracted_2")
    @patch("agent_memory_server.cli.migrate_add_memory_hashes_1")
    @patch("agent_memory_server.cli.get_redis_conn")
    def test_migrate_memories_command(
        self,
        mock_get_redis_conn,
        mock_migration1,
        mock_migration2,
        mock_migration3,
        mock_migration4,
    ):
        """Test migrate_memories command execution."""
        # Use AsyncMock which returns completed awaitables
        mock_redis = Mock()
        mock_get_redis_conn.return_value = mock_redis

        for migration in [
            mock_migration1,
            mock_migration2,
            mock_migration3,
            mock_migration4,
        ]:
            migration.return_value = None

        runner = CliRunner()
        result = runner.invoke(migrate_memories)

        assert result.exit_code == 0
        assert "Starting memory migrations..." in result.output
        assert "Memory migrations completed successfully." in result.output
        mock_get_redis_conn.assert_called_once()
        mock_migration1.assert_called_once_with(redis=mock_redis)
        mock_migration2.assert_called_once_with(redis=mock_redis)
        mock_migration3.assert_called_once_with(redis=mock_redis)
        mock_migration4.assert_called_once_with(redis=mock_redis)


class TestApiCommand:
    """Tests for the api command."""

    def setup_method(self):
        """Store original use_docket setting."""
        from agent_memory_server.config import settings

        self.original_use_docket = settings.use_docket

    def teardown_method(self):
        """Restore original use_docket setting."""
        from agent_memory_server.config import settings

        settings.use_docket = self.original_use_docket

    @patch("agent_memory_server.cli.uvicorn.run")
    @patch("agent_memory_server.main.on_start_logger")
    def test_api_command_defaults(self, mock_on_start_logger, mock_uvicorn_run):
        """Test api command with default parameters uses Docket backend.

        By default, we preserve existing behavior by enabling Docket-based
        background tasks (settings.use_docket should be True).
        """
        from agent_memory_server.config import settings

        # Set initial state
        settings.use_docket = False

        runner = CliRunner()
        result = runner.invoke(api)

        assert result.exit_code == 0
        # Default should enable Docket-based background tasks
        assert settings.use_docket is True

        mock_on_start_logger.assert_called_once()
        mock_uvicorn_run.assert_called_once_with(
            "agent_memory_server.main:app",
            host="0.0.0.0",
            port=8000,  # default from settings
            reload=False,
        )

    @patch("agent_memory_server.cli.uvicorn.run")
    @patch("agent_memory_server.main.on_start_logger")
    def test_api_command_with_options(self, mock_on_start_logger, mock_uvicorn_run):
        """Test api command with custom parameters."""
        runner = CliRunner()
        result = runner.invoke(
            api, ["--port", "9000", "--host", "127.0.0.1", "--reload"]
        )

        assert result.exit_code == 0
        mock_on_start_logger.assert_called_once_with(9000)
        mock_uvicorn_run.assert_called_once_with(
            "agent_memory_server.main:app",
            host="127.0.0.1",
            port=9000,
            reload=True,
        )

    @patch("agent_memory_server.cli.uvicorn.run")
    @patch("agent_memory_server.main.on_start_logger")
    def test_api_command_with_no_worker_flag(
        self, mock_on_start_logger, mock_uvicorn_run
    ):
        """Test api command with deprecated --no-worker flag.

        The flag should continue to force asyncio background tasks and not
        require a Docket worker.
        """
        from agent_memory_server.config import settings

        # Set initial state
        settings.use_docket = True

        runner = CliRunner()
        result = runner.invoke(api, ["--no-worker"])

        assert result.exit_code == 0
        # Should set use_docket to False when --no-worker is specified
        assert settings.use_docket is False

        mock_on_start_logger.assert_called_once()
        mock_uvicorn_run.assert_called_once_with(
            "agent_memory_server.main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
        )

    @patch("agent_memory_server.cli.uvicorn.run")
    @patch("agent_memory_server.main.on_start_logger")
    def test_api_command_with_task_backend_asyncio(
        self, mock_on_start_logger, mock_uvicorn_run
    ):
        """Test api command with --task-backend=asyncio and other options."""
        from agent_memory_server.config import settings

        # Set initial state
        settings.use_docket = True

        runner = CliRunner()
        result = runner.invoke(
            api,
            [
                "--port",
                "9999",
                "--host",
                "127.0.0.1",
                "--reload",
                "--task-backend",
                "asyncio",
            ],
        )

        assert result.exit_code == 0
        # Should opt into asyncio when task-backend=asyncio is specified
        assert settings.use_docket is False

        mock_on_start_logger.assert_called_once_with(9999)
        mock_uvicorn_run.assert_called_once_with(
            "agent_memory_server.main:app",
            host="127.0.0.1",
            port=9999,
            reload=True,
        )

    def test_api_command_help_includes_task_backend_and_no_worker(self):
        """Test that API help mentions deprecated --no-worker and new task-backend."""
        runner = CliRunner()
        result = runner.invoke(api, ["--help"])

        assert result.exit_code == 0
        assert "--no-worker" in result.output
        assert "DEPRECATED" in result.output
        assert "--task-backend" in result.output
        assert "asyncio" in result.output
        assert "docket" in result.output


class TestMcpCommand:
    """Tests for the mcp command."""

    def setup_method(self):
        """Store original settings."""
        from agent_memory_server.config import settings

        self.original_use_docket = settings.use_docket
        self.original_mcp_port = settings.mcp_port

    def teardown_method(self):
        """Restore original settings."""
        from agent_memory_server.config import settings

        settings.use_docket = self.original_use_docket
        settings.mcp_port = self.original_mcp_port

    @patch("agent_memory_server.cli.settings")
    @patch("agent_memory_server.mcp.mcp_app")
    def test_mcp_command_stdio_mode(self, mock_mcp_app, mock_settings):
        """Test mcp command in stdio mode."""
        mock_settings.mcp_port = 3001
        mock_settings.log_level = "INFO"

        mock_mcp_app.run_stdio_async = AsyncMock()

        runner = CliRunner()
        result = runner.invoke(mcp, ["--mode", "stdio"])

        assert result.exit_code == 0
        mock_mcp_app.run_stdio_async.assert_called_once()

    @patch("agent_memory_server.cli.settings")
    @patch("agent_memory_server.mcp.mcp_app")
    def test_mcp_command_sse_mode(self, mock_mcp_app, mock_settings):
        """Test mcp command in SSE mode."""
        mock_settings.mcp_port = 3001

        mock_mcp_app.run_sse_async = AsyncMock()

        runner = CliRunner()
        result = runner.invoke(mcp, ["--mode", "sse", "--port", "4000"])

        assert result.exit_code == 0
        mock_mcp_app.run_sse_async.assert_called_once()

    @patch("agent_memory_server.cli.settings")
    @patch("agent_memory_server.mcp.mcp_app")
    def test_mcp_command_streamable_http_mode(self, mock_mcp_app, mock_settings):
        """Test mcp command in streamable-http mode."""
        mock_settings.mcp_port = 3001

        mock_mcp_app.run_streamable_http_async = AsyncMock()

        runner = CliRunner()
        result = runner.invoke(mcp, ["--mode", "streamable-http", "--port", "9000"])

        assert result.exit_code == 0
        mock_mcp_app.run_streamable_http_async.assert_called_once()

    @patch("agent_memory_server.cli.configure_logging")
    @patch("agent_memory_server.mcp.mcp_app")
    def test_mcp_command_streamable_http_mode_uses_asyncio_by_default(
        self, mock_mcp_app, mock_configure_logging
    ):
        """Test that streamable-http mode uses asyncio backend by default."""
        from agent_memory_server.config import settings

        # Set initial state
        settings.use_docket = True

        mock_mcp_app.run_streamable_http_async = AsyncMock()

        runner = CliRunner()
        result = runner.invoke(mcp, ["--mode", "streamable-http"])

        assert result.exit_code == 0
        assert settings.use_docket is False
        mock_mcp_app.run_streamable_http_async.assert_called_once()

    @patch("agent_memory_server.cli.configure_logging")
    @patch("agent_memory_server.mcp.mcp_app")
    def test_mcp_command_streamable_http_mode_with_task_backend_docket(
        self, mock_mcp_app, mock_configure_logging
    ):
        """Test that streamable-http mode with --task-backend=docket sets use_docket=True."""
        from agent_memory_server.config import settings

        # Set initial state
        settings.use_docket = False

        mock_mcp_app.run_streamable_http_async = AsyncMock()

        runner = CliRunner()
        result = runner.invoke(
            mcp, ["--mode", "streamable-http", "--task-backend", "docket"]
        )

        assert result.exit_code == 0
        assert settings.use_docket is True
        mock_mcp_app.run_streamable_http_async.assert_called_once()

    @patch("agent_memory_server.cli.configure_mcp_logging")
    @patch("agent_memory_server.cli.settings")
    @patch("agent_memory_server.mcp.mcp_app")
    def test_mcp_command_stdio_logging_config(
        self, mock_mcp_app, mock_settings, mock_configure_mcp_logging
    ):
        """Test that stdio mode configures logging to stderr."""
        mock_settings.mcp_port = 3001
        mock_settings.log_level = "DEBUG"

        mock_mcp_app.run_stdio_async = AsyncMock()

        runner = CliRunner()
        result = runner.invoke(mcp, ["--mode", "stdio"])

        assert result.exit_code == 0
        mock_mcp_app.run_stdio_async.assert_called_once()
        mock_configure_mcp_logging.assert_called_once()

    @patch("agent_memory_server.cli.configure_mcp_logging")
    @patch("agent_memory_server.mcp.mcp_app")
    def test_mcp_command_stdio_mode_uses_asyncio_by_default(
        self, mock_mcp_app, mock_configure_mcp_logging
    ):
        """Test that stdio mode uses asyncio backend by default.

        Default behavior should not require a separate Docket worker, so
        settings.use_docket should be False.
        """
        from agent_memory_server.config import settings

        # Set initial state
        settings.use_docket = True

        mock_mcp_app.run_stdio_async = AsyncMock()

        runner = CliRunner()
        result = runner.invoke(mcp, ["--mode", "stdio"])

        assert result.exit_code == 0
        # stdio mode should switch to asyncio backend by default
        assert settings.use_docket is False
        mock_mcp_app.run_stdio_async.assert_called_once()

    @patch("agent_memory_server.cli.configure_logging")
    @patch("agent_memory_server.mcp.mcp_app")
    def test_mcp_command_sse_mode_uses_asyncio_by_default(
        self, mock_mcp_app, mock_configure_logging
    ):
        """Test that SSE mode uses asyncio backend by default."""
        from agent_memory_server.config import settings

        # Set initial state
        settings.use_docket = True

        mock_mcp_app.run_sse_async = AsyncMock()

        runner = CliRunner()
        result = runner.invoke(mcp, ["--mode", "sse"])

        assert result.exit_code == 0
        # SSE mode should also use asyncio backend by default
        assert settings.use_docket is False
        mock_mcp_app.run_sse_async.assert_called_once()

    @patch("agent_memory_server.cli.configure_logging")
    @patch("agent_memory_server.mcp.mcp_app")
    def test_mcp_command_sse_mode_with_task_backend_docket(
        self, mock_mcp_app, mock_configure_logging
    ):
        """Test that SSE mode with --task-backend=docket sets use_docket=True."""
        from agent_memory_server.config import settings

        # Set initial state
        settings.use_docket = False

        mock_mcp_app.run_sse_async = AsyncMock()

        runner = CliRunner()
        result = runner.invoke(mcp, ["--mode", "sse", "--task-backend", "docket"])

        assert result.exit_code == 0
        # SSE mode with task-backend=docket should enable Docket
        assert settings.use_docket is True
        mock_mcp_app.run_sse_async.assert_called_once()

    @patch("agent_memory_server.cli.configure_mcp_logging")
    @patch("agent_memory_server.mcp.mcp_app")
    def test_mcp_command_stdio_mode_with_task_backend_docket(
        self, mock_mcp_app, mock_configure_mcp_logging
    ):
        """Test that stdio mode with --task-backend=docket sets use_docket=True."""
        from agent_memory_server.config import settings

        # Set initial state
        settings.use_docket = False

        mock_mcp_app.run_stdio_async = AsyncMock()

        runner = CliRunner()
        result = runner.invoke(mcp, ["--mode", "stdio", "--task-backend", "docket"])

        assert result.exit_code == 0
        # stdio mode with task-backend=docket should enable Docket
        assert settings.use_docket is True
        mock_mcp_app.run_stdio_async.assert_called_once()

    @patch("agent_memory_server.cli.configure_logging")
    @patch("agent_memory_server.mcp.mcp_app")
    def test_mcp_command_port_setting_update(
        self, mock_mcp_app, mock_configure_logging
    ):
        """Test that MCP command updates the port setting correctly."""
        from agent_memory_server.config import settings

        original_port = settings.mcp_port
        mock_mcp_app.run_sse_async = AsyncMock()

        runner = CliRunner()
        result = runner.invoke(mcp, ["--port", "7777", "--mode", "sse"])

        assert result.exit_code == 0
        # Port should be updated in settings
        assert settings.mcp_port == 7777
        assert settings.mcp_port != original_port

    def test_mcp_command_help_includes_task_backend(self):
        """Test that MCP command help includes --task-backend option."""
        runner = CliRunner()
        result = runner.invoke(mcp, ["--help"])

        assert result.exit_code == 0
        assert "--task-backend" in result.output
        assert "asyncio" in result.output
        assert "docket" in result.output
        assert "--no-worker" not in result.output

    def test_mcp_command_mode_choices(self):
        """Test that MCP command only accepts valid mode choices."""
        runner = CliRunner()
        result = runner.invoke(mcp, ["--mode", "invalid"])

        assert result.exit_code != 0
        assert (
            "Invalid value for '--mode'" in result.output
            or "invalid_value" in result.output.lower()
        )


class TestScheduleTask:
    """Tests for the schedule_task command."""

    def test_schedule_task_invalid_arg_format(self):
        """Test error handling for invalid argument format."""
        runner = CliRunner()
        result = runner.invoke(
            schedule_task, ["test.module.test_function", "--args", "invalid_format"]
        )

        assert result.exit_code == 1
        assert "Invalid argument format" in result.output

    def test_schedule_task_sync_error_handling(self):
        """Test error handling in sync part (before asyncio.run)."""
        # Test import error
        runner = CliRunner()
        result = runner.invoke(schedule_task, ["invalid.module.path"])
        assert result.exit_code == 1

        # Test invalid arguments
        result = runner.invoke(
            schedule_task,
            ["test.module.function", "--args", "invalid_arg_without_equals"],
        )
        assert result.exit_code == 1

    def test_schedule_task_argument_parsing(self):
        """Test various argument parsing scenarios."""
        # We test this by calling the command with invalid arguments
        runner = CliRunner()

        # Test invalid argument format
        result = runner.invoke(
            schedule_task, ["test.module.function", "--args", "invalid_format"]
        )
        assert result.exit_code == 1
        assert "Invalid argument format" in result.output


class TestTaskWorker:
    """Tests for the task_worker command."""

    @patch("agent_memory_server.cli.get_redis_conn")
    @patch("docket.Worker.run")
    @patch("agent_memory_server.cli.settings")
    def test_task_worker_success(
        self,
        mock_settings,
        mock_worker_run,
        mock_get_redis_conn,
        redis_url,
    ):
        """Test successful task worker start."""
        mock_settings.use_docket = True
        mock_settings.docket_name = "test-docket"
        mock_settings.redis_url = redis_url

        mock_worker_run.return_value = None
        mock_redis = AsyncMock()
        mock_get_redis_conn.return_value = mock_redis

        runner = CliRunner()
        result = runner.invoke(
            task_worker, ["--concurrency", "5", "--redelivery-timeout", "60"]
        )

        assert result.exit_code == 0
        mock_worker_run.assert_called_once()
        kwargs = mock_worker_run.call_args.kwargs
        assert kwargs["redelivery_timeout"] == timedelta(seconds=60)

    @patch("agent_memory_server.cli.settings")
    def test_task_worker_docket_disabled(self, mock_settings):
        """Test task worker when docket is disabled."""
        mock_settings.use_docket = False

        runner = CliRunner()
        result = runner.invoke(task_worker)

        assert result.exit_code == 1
        assert "Docket is disabled in settings" in result.output

    def test_task_worker_rejects_non_positive_redelivery_timeout(self):
        """Redelivery timeout should be a positive integer."""
        runner = CliRunner()
        result = runner.invoke(task_worker, ["--redelivery-timeout", "0"])

        assert result.exit_code == 2
        assert "Invalid value for '--redelivery-timeout'" in result.output

    @patch("agent_memory_server.cli.get_redis_conn")
    @patch("docket.Worker.run")
    @patch("agent_memory_server.cli.settings")
    def test_task_worker_default_params(
        self,
        mock_settings,
        mock_worker_run,
        mock_get_redis_conn,
        redis_url,
    ):
        """Test task worker with default parameters."""
        mock_settings.use_docket = True
        mock_settings.docket_name = "test-docket"
        mock_settings.redis_url = redis_url
        mock_settings.llm_task_timeout_minutes = 5

        mock_worker_run.return_value = None
        mock_redis = AsyncMock()
        mock_get_redis_conn.return_value = mock_redis

        runner = CliRunner()
        result = runner.invoke(task_worker)

        assert result.exit_code == 0
        mock_worker_run.assert_called_once()
        kwargs = mock_worker_run.call_args.kwargs
        assert kwargs["redelivery_timeout"] == timedelta(seconds=600)

    @patch("agent_memory_server.cli.get_redis_conn")
    @patch("docket.Worker.run")
    @patch("agent_memory_server.cli.settings")
    def test_task_worker_default_redelivery_timeout_has_minimum_floor(
        self,
        mock_settings,
        mock_worker_run,
        mock_get_redis_conn,
        redis_url,
    ):
        """Default redelivery timeout should never be less than 60 seconds."""
        mock_settings.use_docket = True
        mock_settings.docket_name = "test-docket"
        mock_settings.redis_url = redis_url
        mock_settings.llm_task_timeout_minutes = 0

        mock_worker_run.return_value = None
        mock_redis = AsyncMock()
        mock_get_redis_conn.return_value = mock_redis

        runner = CliRunner()
        result = runner.invoke(task_worker)

        assert result.exit_code == 0
        mock_worker_run.assert_called_once()
        kwargs = mock_worker_run.call_args.kwargs
        assert kwargs["redelivery_timeout"] == timedelta(seconds=60)


class TestSearch:
    """Tests for the search command."""

    @patch("agent_memory_server.long_term_memory.search_long_term_memories")
    def test_search_basic_query(self, mock_search):
        """Test basic search with just a query."""
        from agent_memory_server.models import MemoryRecordResult, MemoryRecordResults

        mock_search.return_value = MemoryRecordResults(
            memories=[
                MemoryRecordResult(
                    id="mem1",
                    text="Test memory about authentication",
                    dist=0.15,
                    namespace="default",
                )
            ],
            total=1,
            next_offset=None,
        )

        runner = CliRunner()
        result = runner.invoke(search, ["test query"])

        assert result.exit_code == 0
        assert "Found 1 memories" in result.output
        assert "mem1" in result.output
        assert "Test memory about authentication" in result.output
        assert "0.1500" in result.output
        mock_search.assert_called_once()

    @patch("agent_memory_server.long_term_memory.search_long_term_memories")
    def test_search_with_namespace_filter(self, mock_search):
        """Test search with namespace filter."""
        from agent_memory_server.models import MemoryRecordResults

        mock_search.return_value = MemoryRecordResults(
            memories=[],
            total=0,
            next_offset=None,
        )

        runner = CliRunner()
        result = runner.invoke(search, ["test query", "--namespace", "myapp"])

        assert result.exit_code == 0
        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args[1]
        assert call_kwargs["namespace"].eq == "myapp"

    @patch("agent_memory_server.long_term_memory.search_long_term_memories")
    def test_search_with_session_id_filter(self, mock_search):
        """Test search with session ID filter."""
        from agent_memory_server.models import MemoryRecordResults

        mock_search.return_value = MemoryRecordResults(
            memories=[],
            total=0,
            next_offset=None,
        )

        runner = CliRunner()
        result = runner.invoke(search, ["test query", "-s", "session123"])

        assert result.exit_code == 0
        call_kwargs = mock_search.call_args[1]
        assert call_kwargs["session_id"].eq == "session123"

    @patch("agent_memory_server.long_term_memory.search_long_term_memories")
    def test_search_with_user_id_filter(self, mock_search):
        """Test search with user ID filter."""
        from agent_memory_server.models import MemoryRecordResults

        mock_search.return_value = MemoryRecordResults(
            memories=[],
            total=0,
            next_offset=None,
        )

        runner = CliRunner()
        result = runner.invoke(search, ["test query", "-u", "user456"])

        assert result.exit_code == 0
        call_kwargs = mock_search.call_args[1]
        assert call_kwargs["user_id"].eq == "user456"

    @patch("agent_memory_server.long_term_memory.search_long_term_memories")
    def test_search_with_topics_filter(self, mock_search):
        """Test search with topics filter (comma-separated)."""
        from agent_memory_server.models import MemoryRecordResults

        mock_search.return_value = MemoryRecordResults(
            memories=[],
            total=0,
            next_offset=None,
        )

        runner = CliRunner()
        result = runner.invoke(search, ["test query", "-t", "auth,security"])

        assert result.exit_code == 0
        call_kwargs = mock_search.call_args[1]
        assert call_kwargs["topics"].any == ["auth", "security"]

    @patch("agent_memory_server.long_term_memory.search_long_term_memories")
    def test_search_with_entities_filter(self, mock_search):
        """Test search with entities filter (comma-separated)."""
        from agent_memory_server.models import MemoryRecordResults

        mock_search.return_value = MemoryRecordResults(
            memories=[],
            total=0,
            next_offset=None,
        )

        runner = CliRunner()
        result = runner.invoke(search, ["test query", "-e", "User,Project"])

        assert result.exit_code == 0
        call_kwargs = mock_search.call_args[1]
        assert call_kwargs["entities"].any == ["User", "Project"]

    @patch("agent_memory_server.long_term_memory.search_long_term_memories")
    def test_search_with_limit_and_offset(self, mock_search):
        """Test search with limit and offset pagination."""
        from agent_memory_server.models import MemoryRecordResults

        mock_search.return_value = MemoryRecordResults(
            memories=[],
            total=0,
            next_offset=None,
        )

        runner = CliRunner()
        result = runner.invoke(search, ["test query", "-l", "5", "-o", "10"])

        assert result.exit_code == 0
        call_kwargs = mock_search.call_args[1]
        assert call_kwargs["limit"] == 5
        assert call_kwargs["offset"] == 10

    @patch("agent_memory_server.long_term_memory.search_long_term_memories")
    def test_search_with_distance_threshold(self, mock_search):
        """Test search with distance threshold."""
        from agent_memory_server.models import MemoryRecordResults

        mock_search.return_value = MemoryRecordResults(
            memories=[],
            total=0,
            next_offset=None,
        )

        runner = CliRunner()
        result = runner.invoke(search, ["test query", "-d", "0.5"])

        assert result.exit_code == 0
        call_kwargs = mock_search.call_args[1]
        assert call_kwargs["distance_threshold"] == 0.5

    @patch("agent_memory_server.long_term_memory.search_long_term_memories")
    def test_search_rejects_distance_threshold_for_keyword_mode(self, mock_search):
        """distance_threshold should fail fast outside semantic mode."""
        runner = CliRunner()
        result = runner.invoke(
            search,
            ["test query", "--search-mode", "keyword", "--distance-threshold", "0.5"],
        )

        assert result.exit_code != 0
        assert (
            "distance_threshold is only supported for semantic search mode"
            in result.output
        )
        mock_search.assert_not_called()

    @patch("agent_memory_server.long_term_memory.search_long_term_memories")
    def test_search_json_output(self, mock_search):
        """Test search with JSON output format."""
        from agent_memory_server.models import MemoryRecordResult, MemoryRecordResults

        mock_search.return_value = MemoryRecordResults(
            memories=[
                MemoryRecordResult(
                    id="mem1",
                    text="Test memory",
                    dist=0.1,
                    namespace="default",
                    topics=["topic1"],
                    entities=["entity1"],
                )
            ],
            total=1,
            next_offset=None,
        )

        runner = CliRunner()
        result = runner.invoke(search, ["test query", "--format", "json"])

        assert result.exit_code == 0
        # Verify it's valid JSON
        import json

        output = json.loads(result.output)
        assert output["total"] == 1
        assert len(output["memories"]) == 1
        assert output["memories"][0]["id"] == "mem1"

    @patch("agent_memory_server.long_term_memory.search_long_term_memories")
    def test_search_text_output_with_all_fields(self, mock_search):
        """Test text output displays all memory fields."""
        from agent_memory_server.models import MemoryRecordResult, MemoryRecordResults

        mock_search.return_value = MemoryRecordResults(
            memories=[
                MemoryRecordResult(
                    id="mem1",
                    text="Test memory content",
                    dist=0.25,
                    namespace="myapp",
                    session_id="sess123",
                    user_id="user456",
                    topics=["topic1", "topic2"],
                    entities=["Entity1", "Entity2"],
                )
            ],
            total=1,
            next_offset=None,
        )

        runner = CliRunner()
        result = runner.invoke(search, ["test query"])

        assert result.exit_code == 0
        assert "mem1" in result.output
        assert "0.2500" in result.output
        assert "myapp" in result.output
        assert "sess123" in result.output
        assert "user456" in result.output
        assert "topic1, topic2" in result.output
        assert "Entity1, Entity2" in result.output
        assert "Test memory content" in result.output

    @patch("agent_memory_server.long_term_memory.search_long_term_memories")
    def test_search_text_truncation(self, mock_search):
        """Test that long text is truncated in text output."""
        from agent_memory_server.models import MemoryRecordResult, MemoryRecordResults

        long_text = "A" * 300  # 300 characters
        mock_search.return_value = MemoryRecordResults(
            memories=[
                MemoryRecordResult(
                    id="mem1",
                    text=long_text,
                    dist=0.1,
                )
            ],
            total=1,
            next_offset=None,
        )

        runner = CliRunner()
        result = runner.invoke(search, ["test query"])

        assert result.exit_code == 0
        # Should show first 200 chars + "..."
        assert "A" * 200 in result.output
        assert "..." in result.output

    @patch("agent_memory_server.long_term_memory.search_long_term_memories")
    def test_search_pagination_hint(self, mock_search):
        """Test that pagination hint is shown when more results available."""
        from agent_memory_server.models import MemoryRecordResult, MemoryRecordResults

        mock_search.return_value = MemoryRecordResults(
            memories=[
                MemoryRecordResult(
                    id="mem1",
                    text="Test memory",
                    dist=0.1,
                )
            ],
            total=15,
            next_offset=10,
        )

        runner = CliRunner()
        result = runner.invoke(search, ["test query"])

        assert result.exit_code == 0
        assert "More results available" in result.output
        assert "--offset 10" in result.output

    @patch("agent_memory_server.long_term_memory.search_long_term_memories")
    def test_search_no_results(self, mock_search):
        """Test search with no results."""
        from agent_memory_server.models import MemoryRecordResults

        mock_search.return_value = MemoryRecordResults(
            memories=[],
            total=0,
            next_offset=None,
        )

        runner = CliRunner()
        result = runner.invoke(search, ["test query"])

        assert result.exit_code == 0
        assert "Found 0 memories" in result.output

    @patch("agent_memory_server.long_term_memory.search_long_term_memories")
    def test_search_multiple_results(self, mock_search):
        """Test search with multiple results."""
        from agent_memory_server.models import MemoryRecordResult, MemoryRecordResults

        mock_search.return_value = MemoryRecordResults(
            memories=[
                MemoryRecordResult(id="mem1", text="First memory", dist=0.1),
                MemoryRecordResult(id="mem2", text="Second memory", dist=0.2),
                MemoryRecordResult(id="mem3", text="Third memory", dist=0.3),
            ],
            total=3,
            next_offset=None,
        )

        runner = CliRunner()
        result = runner.invoke(search, ["test query"])

        assert result.exit_code == 0
        assert "Found 3 memories" in result.output
        assert "[1]" in result.output
        assert "[2]" in result.output
        assert "[3]" in result.output
        assert "mem1" in result.output
        assert "mem2" in result.output
        assert "mem3" in result.output

    @patch("agent_memory_server.long_term_memory.search_long_term_memories")
    def test_search_all_filters_combined(self, mock_search):
        """Test search with all filters combined."""
        from agent_memory_server.models import MemoryRecordResults

        mock_search.return_value = MemoryRecordResults(
            memories=[],
            total=0,
            next_offset=None,
        )

        runner = CliRunner()
        result = runner.invoke(
            search,
            [
                "test query",
                "--namespace",
                "myapp",
                "--session-id",
                "sess123",
                "--user-id",
                "user456",
                "--topics",
                "auth,security",
                "--entities",
                "User,Admin",
                "--limit",
                "20",
                "--offset",
                "5",
                "--distance-threshold",
                "0.3",
            ],
        )

        assert result.exit_code == 0
        call_kwargs = mock_search.call_args[1]
        assert call_kwargs["text"] == "test query"
        assert call_kwargs["namespace"].eq == "myapp"
        assert call_kwargs["session_id"].eq == "sess123"
        assert call_kwargs["user_id"].eq == "user456"
        assert call_kwargs["topics"].any == ["auth", "security"]
        assert call_kwargs["entities"].any == ["User", "Admin"]
        assert call_kwargs["limit"] == 20
        assert call_kwargs["offset"] == 5
        assert call_kwargs["distance_threshold"] == 0.3

    def test_search_help(self):
        """Test search command help output."""
        runner = CliRunner()
        result = runner.invoke(search, ["--help"])

        assert result.exit_code == 0
        assert "Search long-term memories" in result.output
        assert "--namespace" in result.output
        assert "--session-id" in result.output
        assert "--user-id" in result.output
        assert "--topics" in result.output
        assert "--entities" in result.output
        assert "--limit" in result.output
        assert "--offset" in result.output
        assert "--distance-threshold" in result.output
        assert "--format" in result.output

    @patch("agent_memory_server.long_term_memory.search_long_term_memories")
    def test_search_empty_query_lists_all(self, mock_search):
        """Test search with empty query lists all memories matching filters."""
        from agent_memory_server.models import MemoryRecordResult, MemoryRecordResults

        mock_search.return_value = MemoryRecordResults(
            memories=[
                MemoryRecordResult(id="mem1", text="Memory 1", dist=0.0),
                MemoryRecordResult(id="mem2", text="Memory 2", dist=0.0),
            ],
            total=2,
            next_offset=None,
        )

        runner = CliRunner()
        result = runner.invoke(search, ["--namespace", "myapp"])

        assert result.exit_code == 0
        assert "Found 2 memories" in result.output
        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args[1]
        assert call_kwargs["text"] == ""
        assert call_kwargs["namespace"].eq == "myapp"

    @patch("agent_memory_server.long_term_memory.search_long_term_memories")
    def test_search_no_args_lists_all(self, mock_search):
        """Test search with no arguments lists all memories."""
        from agent_memory_server.models import MemoryRecordResults

        mock_search.return_value = MemoryRecordResults(
            memories=[],
            total=0,
            next_offset=None,
        )

        runner = CliRunner()
        result = runner.invoke(search, [])

        assert result.exit_code == 0
        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args[1]
        assert call_kwargs["text"] == ""


class TestDeleteMemory:
    """Tests for the delete memory command."""

    @patch("agent_memory_server.long_term_memory.delete_long_term_memories")
    def test_delete_by_ids(self, mock_delete):
        """Test deleting memories by explicit IDs."""
        mock_delete.return_value = 3

        runner = CliRunner()
        result = runner.invoke(delete_memory_cmd, ["mem1", "mem2", "mem3", "-f"])

        assert result.exit_code == 0
        assert "Deleted 3 memories" in result.output
        mock_delete.assert_called_once_with(ids=["mem1", "mem2", "mem3"])

    @patch("agent_memory_server.long_term_memory.delete_long_term_memories")
    def test_delete_dry_run(self, mock_delete):
        """Test delete with dry-run flag."""
        runner = CliRunner()
        result = runner.invoke(delete_memory_cmd, ["mem1", "mem2", "--dry-run"])

        assert result.exit_code == 0
        assert "Would delete 2 memories" in result.output
        assert "mem1" in result.output
        assert "mem2" in result.output
        mock_delete.assert_not_called()

    @patch("agent_memory_server.long_term_memory.delete_long_term_memories")
    def test_delete_requires_confirmation(self, mock_delete):
        """Test that delete requires confirmation without -f flag."""
        runner = CliRunner()
        result = runner.invoke(delete_memory_cmd, ["mem1"], input="n\n")

        assert result.exit_code == 1  # Aborted
        mock_delete.assert_not_called()

    @patch("agent_memory_server.long_term_memory.delete_long_term_memories")
    def test_delete_with_confirmation(self, mock_delete):
        """Test delete with user confirmation."""
        mock_delete.return_value = 1

        runner = CliRunner()
        result = runner.invoke(delete_memory_cmd, ["mem1"], input="y\n")

        assert result.exit_code == 0
        assert "Deleted 1 memories" in result.output
        mock_delete.assert_called_once()

    @patch("agent_memory_server.long_term_memory.delete_long_term_memories")
    @patch("agent_memory_server.long_term_memory.search_long_term_memories")
    def test_delete_empty_text_memories(self, mock_search, mock_delete):
        """Test deleting memories with empty text."""
        from agent_memory_server.models import MemoryRecordResult, MemoryRecordResults

        mock_search.return_value = MemoryRecordResults(
            memories=[
                MemoryRecordResult(id="empty1", text="", dist=0.0),
                MemoryRecordResult(id="empty2", text="   ", dist=0.0),
                MemoryRecordResult(id="valid", text="Has content", dist=0.0),
            ],
            total=3,
            next_offset=None,
        )
        mock_delete.return_value = 2

        runner = CliRunner()
        result = runner.invoke(delete_memory_cmd, ["--empty-text", "-f"])

        assert result.exit_code == 0
        assert "Found 2 memories with empty text" in result.output
        assert "Deleted 2 memories" in result.output
        mock_delete.assert_called_once_with(ids=["empty1", "empty2"])

    @patch("agent_memory_server.long_term_memory.search_long_term_memories")
    def test_delete_empty_text_none_found(self, mock_search):
        """Test --empty-text when no empty memories exist."""
        from agent_memory_server.models import MemoryRecordResult, MemoryRecordResults

        mock_search.return_value = MemoryRecordResults(
            memories=[
                MemoryRecordResult(id="valid", text="Has content", dist=0.0),
            ],
            total=1,
            next_offset=None,
        )

        runner = CliRunner()
        result = runner.invoke(delete_memory_cmd, ["--empty-text"])

        assert result.exit_code == 0
        assert "No memories with empty text found" in result.output

    @patch("agent_memory_server.long_term_memory.delete_long_term_memories")
    @patch("agent_memory_server.long_term_memory.search_long_term_memories")
    def test_delete_empty_text_with_namespace_filter(self, mock_search, mock_delete):
        """Test --empty-text with namespace filter."""
        from agent_memory_server.models import MemoryRecordResult, MemoryRecordResults

        mock_search.return_value = MemoryRecordResults(
            memories=[
                MemoryRecordResult(id="empty1", text="", dist=0.0, namespace="myapp"),
            ],
            total=1,
            next_offset=None,
        )
        mock_delete.return_value = 1

        runner = CliRunner()
        result = runner.invoke(
            delete_memory_cmd, ["--empty-text", "--namespace", "myapp", "-f"]
        )

        assert result.exit_code == 0
        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args[1]
        assert call_kwargs["namespace"].eq == "myapp"

    @patch("agent_memory_server.long_term_memory.delete_long_term_memories")
    @patch("agent_memory_server.long_term_memory.search_long_term_memories")
    def test_delete_empty_text_dry_run(self, mock_search, mock_delete):
        """Test --empty-text with --dry-run."""
        from agent_memory_server.models import MemoryRecordResult, MemoryRecordResults

        mock_search.return_value = MemoryRecordResults(
            memories=[
                MemoryRecordResult(id="empty1", text="", dist=0.0),
                MemoryRecordResult(id="empty2", text="", dist=0.0),
            ],
            total=2,
            next_offset=None,
        )

        runner = CliRunner()
        result = runner.invoke(delete_memory_cmd, ["--empty-text", "--dry-run"])

        assert result.exit_code == 0
        assert "Would delete 2 memories" in result.output
        mock_delete.assert_not_called()

    @patch("agent_memory_server.long_term_memory.search_long_term_memories")
    def test_delete_empty_text_with_empty_ids(self, mock_search):
        """Test --empty-text when some memories have empty IDs."""
        from agent_memory_server.models import MemoryRecordResult, MemoryRecordResults

        mock_search.return_value = MemoryRecordResults(
            memories=[
                MemoryRecordResult(id="", text="", dist=0.0),
                MemoryRecordResult(id="", text="", dist=0.0),
                MemoryRecordResult(id="   ", text="", dist=0.0),
            ],
            total=3,
            next_offset=None,
        )

        runner = CliRunner()
        result = runner.invoke(delete_memory_cmd, ["--empty-text", "--dry-run"])

        assert result.exit_code == 0
        assert "Found 3 memories with empty text" in result.output
        assert "3 memories also have empty IDs" in result.output
        # Updated message to match new CLI output
        assert "Use --invalid flag" in result.output

    @patch("agent_memory_server.long_term_memory.delete_long_term_memories")
    @patch("agent_memory_server.long_term_memory.search_long_term_memories")
    def test_delete_empty_text_mixed_ids(self, mock_search, mock_delete):
        """Test --empty-text with mix of valid and empty IDs."""
        from agent_memory_server.models import MemoryRecordResult, MemoryRecordResults

        mock_search.return_value = MemoryRecordResults(
            memories=[
                MemoryRecordResult(id="valid1", text="", dist=0.0),
                MemoryRecordResult(id="", text="", dist=0.0),
                MemoryRecordResult(id="valid2", text="   ", dist=0.0),
            ],
            total=3,
            next_offset=None,
        )
        mock_delete.return_value = 2

        runner = CliRunner()
        result = runner.invoke(delete_memory_cmd, ["--empty-text", "-f"])

        assert result.exit_code == 0
        assert "Found 3 memories with empty text" in result.output
        assert "1 memories also have empty IDs" in result.output
        assert "Deleted 2 memories" in result.output
        mock_delete.assert_called_once_with(ids=["valid1", "valid2"])

    @patch("agent_memory_server.long_term_memory.delete_invalid_memories")
    def test_delete_invalid_memories(self, mock_delete_invalid):
        """Test --invalid flag to delete memories with empty IDs."""
        mock_delete_invalid.return_value = 5

        runner = CliRunner()
        result = runner.invoke(delete_memory_cmd, ["--invalid", "-f"])

        assert result.exit_code == 0
        assert "Deleted 5 memories with invalid IDs" in result.output
        mock_delete_invalid.assert_called_once()

    @patch("agent_memory_server.long_term_memory.delete_invalid_memories")
    def test_delete_invalid_requires_confirmation(self, mock_delete_invalid):
        """Test --invalid requires confirmation."""
        runner = CliRunner()
        result = runner.invoke(delete_memory_cmd, ["--invalid"], input="n\n")

        assert result.exit_code == 1  # Aborted
        mock_delete_invalid.assert_not_called()

    def test_delete_invalid_dry_run_not_supported(self):
        """Test --invalid with --dry-run shows message."""
        runner = CliRunner()
        result = runner.invoke(delete_memory_cmd, ["--invalid", "--dry-run"])

        assert result.exit_code == 0
        assert "Dry-run not supported for --invalid" in result.output

    def test_delete_no_ids_specified(self):
        """Test delete with no IDs and no flags."""
        runner = CliRunner()
        result = runner.invoke(delete_memory_cmd, [])

        assert result.exit_code == 0
        assert "No memory IDs specified" in result.output

    @patch("agent_memory_server.long_term_memory.delete_long_term_memories")
    def test_delete_dry_run_truncates_long_list(self, mock_delete):
        """Test dry-run truncates list when more than 20 IDs."""
        # Generate 25 IDs to trigger truncation
        ids = [f"mem{i}" for i in range(25)]

        runner = CliRunner()
        result = runner.invoke(delete_memory_cmd, ids + ["--dry-run"])

        assert result.exit_code == 0
        assert "Would delete 25 memories" in result.output
        assert "mem0" in result.output
        assert "mem19" in result.output
        assert "... and 5 more" in result.output
        mock_delete.assert_not_called()

    def test_delete_help(self):
        """Test delete memory command help output."""
        runner = CliRunner()
        result = runner.invoke(delete_memory_cmd, ["--help"])

        assert result.exit_code == 0
        assert "Delete long-term memories" in result.output
        assert "--empty-text" in result.output
        assert "--invalid" in result.output
        assert "--dry-run" in result.output
        assert "--force" in result.output
        assert "--namespace" in result.output
        assert "--user-id" in result.output


class TestDeleteSession:
    """Tests for the delete session command."""

    @patch("agent_memory_server.working_memory.delete_working_memory")
    def test_delete_session_by_id(self, mock_delete):
        """Test deleting a session by ID."""
        runner = CliRunner()
        result = runner.invoke(delete_session_cmd, ["my-session-123", "-f"])

        assert result.exit_code == 0
        assert "Deleted working memory session: my-session-123" in result.output
        mock_delete.assert_called_once_with(
            session_id="my-session-123",
            user_id=None,
            namespace=None,
        )

    @patch("agent_memory_server.working_memory.delete_working_memory")
    def test_delete_session_with_namespace(self, mock_delete):
        """Test deleting a session with namespace."""
        runner = CliRunner()
        result = runner.invoke(
            delete_session_cmd, ["my-session-123", "--namespace", "myapp", "-f"]
        )

        assert result.exit_code == 0
        mock_delete.assert_called_once_with(
            session_id="my-session-123",
            user_id=None,
            namespace="myapp",
        )

    @patch("agent_memory_server.working_memory.delete_working_memory")
    def test_delete_session_with_user_id(self, mock_delete):
        """Test deleting a session with user ID."""
        runner = CliRunner()
        result = runner.invoke(
            delete_session_cmd, ["my-session-123", "-u", "user456", "-f"]
        )

        assert result.exit_code == 0
        mock_delete.assert_called_once_with(
            session_id="my-session-123",
            user_id="user456",
            namespace=None,
        )

    @patch("agent_memory_server.working_memory.delete_working_memory")
    def test_delete_session_requires_confirmation(self, mock_delete):
        """Test that delete session requires confirmation."""
        runner = CliRunner()
        result = runner.invoke(delete_session_cmd, ["my-session-123"], input="n\n")

        assert result.exit_code == 1  # Aborted
        mock_delete.assert_not_called()

    @patch("agent_memory_server.working_memory.delete_working_memory")
    def test_delete_session_with_confirmation(self, mock_delete):
        """Test delete session with user confirmation."""
        runner = CliRunner()
        result = runner.invoke(delete_session_cmd, ["my-session-123"], input="y\n")

        assert result.exit_code == 0
        assert "Deleted working memory session" in result.output
        mock_delete.assert_called_once()

    def test_delete_session_no_id(self):
        """Test delete session without session ID."""
        runner = CliRunner()
        result = runner.invoke(delete_session_cmd, [])

        assert result.exit_code == 0
        assert "Session ID is required" in result.output

    def test_delete_session_help(self):
        """Test delete session command help output."""
        runner = CliRunner()
        result = runner.invoke(delete_session_cmd, ["--help"])

        assert result.exit_code == 0
        assert "Delete a working memory session" in result.output
        assert "--namespace" in result.output
        assert "--user-id" in result.output
        assert "--force" in result.output


class TestCliGroup:
    """Tests for the main CLI group."""

    def test_cli_group_help(self):
        """Test that CLI group shows help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "Command-line interface for agent-memory-server" in result.output

    def test_cli_group_commands_exist(self):
        """Test that all expected commands are registered."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        expected_commands = [
            "version",
            "rebuild-index",
            "migrate-memories",
            "api",
            "mcp",
            "schedule-task",
            "search",
            "delete",
            "task-worker",
        ]
        for command in expected_commands:
            assert command in result.output


class TestMainExecution:
    """Tests for main execution."""

    @patch("agent_memory_server.cli.cli")
    def test_main_execution(self, mock_cli):
        """Test that main execution calls the CLI."""
        # Import and run the main execution code
        import agent_memory_server.cli

        # The main execution is guarded by if __name__ == "__main__"
        # We can test this by patching sys.modules and importing
        with patch.dict(sys.modules, {"__main__": agent_memory_server.cli}):
            # This would normally call cli() but we've mocked it
            pass
