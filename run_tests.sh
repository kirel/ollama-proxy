#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Running Unit Tests${NC}"
python -m pytest tests/test_api_endpoints.py tests/test_model_mapping.py tests/test_streaming.py -v

# Check if integration tests should be run
if [ "$1" == "--integration" ]; then
  echo -e "\n${YELLOW}Running Integration Tests${NC}"
  echo -e "${YELLOW}Note: Ensure that the Ollama API proxy is running (python run.py)${NC}"
  echo -e "${YELLOW}and you have the necessary API keys configured.${NC}"
  echo -e "${YELLOW}Waiting 3 seconds before starting...${NC}"
  sleep 3
  RUN_INTEGRATION_TESTS=1 python -m pytest tests/test_integration.py -v
else
  echo -e "\n${YELLOW}Skipping Integration Tests${NC}"
  echo -e "Run with --integration flag to include integration tests."
fi
