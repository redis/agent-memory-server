from unittest.mock import MagicMock, patch


class TestAWSUtilities:
    """Test cases for AWS utility functions."""

    def test_bedrock_embedding_model_exists_found(self):
        """Test bedrock_embedding_model_exists returns True when model is found."""
        from agent_memory_server._aws.utils import bedrock_embedding_model_exists

        # Clear the cache for this test
        bedrock_embedding_model_exists.cache_clear()

        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {
                "modelSummaries": [
                    {
                        "modelId": "amazon.titan-embed-text-v2:0",
                        "outputModalities": ["EMBEDDING"],
                    },
                    {
                        "modelId": "amazon.titan-embed-text-v1",
                        "outputModalities": ["EMBEDDING"],
                    },
                ]
            }
        ]

        mock_client = MagicMock()
        mock_client.get_paginator.return_value = mock_paginator

        with patch(
            "agent_memory_server._aws.utils.create_bedrock_client",
            return_value=mock_client,
        ):
            result = bedrock_embedding_model_exists(
                "amazon.titan-embed-text-v2:0",
                region_name="us-east-1",
            )

        assert result is True
        mock_client.get_paginator.assert_called_once_with("list_foundation_models")
        mock_paginator.paginate.assert_called_once_with(model_modality="EMBEDDING")

    def test_bedrock_embedding_model_exists_not_found(self):
        """Test bedrock_embedding_model_exists returns False when model is not found."""
        from agent_memory_server._aws.utils import bedrock_embedding_model_exists

        # Clear the cache for this test
        bedrock_embedding_model_exists.cache_clear()

        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {
                "modelSummaries": [
                    {
                        "modelId": "amazon.titan-embed-text-v1",
                        "outputModalities": ["EMBEDDING"],
                    },
                ]
            }
        ]

        mock_client = MagicMock()
        mock_client.get_paginator.return_value = mock_paginator

        with patch(
            "agent_memory_server._aws.utils.create_bedrock_client",
            return_value=mock_client,
        ):
            result = bedrock_embedding_model_exists(
                "non-existent-model",
                region_name="us-east-1",
            )

        assert result is False

    def test_bedrock_embedding_model_exists_via_model_modality(self):
        """Test model detection via modelModality field."""
        from agent_memory_server._aws.utils import bedrock_embedding_model_exists

        # Clear the cache for this test
        bedrock_embedding_model_exists.cache_clear()

        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {
                "modelSummaries": [
                    {
                        "modelId": "cohere.embed-english-v3",
                        "modelModality": "EMBEDDING",
                        "outputModalities": [],  # Empty but modelModality is EMBEDDING
                    },
                ]
            }
        ]

        mock_client = MagicMock()
        mock_client.get_paginator.return_value = mock_paginator

        with patch(
            "agent_memory_server._aws.utils.create_bedrock_client",
            return_value=mock_client,
        ):
            result = bedrock_embedding_model_exists(
                "cohere.embed-english-v3",
                region_name="us-east-1",
            )

        assert result is True

    def test_bedrock_embedding_model_exists_client_error(self):
        """Test bedrock_embedding_model_exists returns False on ClientError."""
        from botocore.exceptions import ClientError

        from agent_memory_server._aws.utils import bedrock_embedding_model_exists

        # Clear the cache for this test
        bedrock_embedding_model_exists.cache_clear()

        mock_client = MagicMock()
        mock_client.get_paginator.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Access Denied"}},
            "ListFoundationModels",
        )

        with patch(
            "agent_memory_server._aws.utils.create_bedrock_client",
            return_value=mock_client,
        ):
            result = bedrock_embedding_model_exists(
                "amazon.titan-embed-text-v2:0",
                region_name="us-east-1",
            )

        assert result is False

    def test_bedrock_embedding_model_exists_caching(self):
        """Test that results are cached."""
        from agent_memory_server._aws.utils import bedrock_embedding_model_exists

        # Clear the cache for this test
        bedrock_embedding_model_exists.cache_clear()

        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {
                "modelSummaries": [
                    {
                        "modelId": "amazon.titan-embed-text-v2:0",
                        "outputModalities": ["EMBEDDING"],
                    },
                ]
            }
        ]

        mock_client = MagicMock()
        mock_client.get_paginator.return_value = mock_paginator

        with patch(
            "agent_memory_server._aws.utils.create_bedrock_client",
            return_value=mock_client,
        ) as mock_create_client:
            # First call
            result1 = bedrock_embedding_model_exists(
                "amazon.titan-embed-text-v2:0",
                region_name="us-east-1",
            )
            # Second call (should be cached)
            result2 = bedrock_embedding_model_exists(
                "amazon.titan-embed-text-v2:0",
                region_name="us-east-1",
            )

        assert result1 is True
        assert result2 is True
        # Should only create client once due to caching
        assert mock_create_client.call_count == 1
