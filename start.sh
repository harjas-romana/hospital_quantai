#!/bin/bash

# Colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting QuantAI Hospital Application...${NC}"
echo -e "${BLUE}=====================================${NC}"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 is not installed. Please install Python 3 to run the backend server.${NC}"
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo -e "${RED}Node.js is not installed. Please install Node.js to run the frontend server.${NC}"
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo -e "${RED}npm is not installed. Please install npm to run the frontend server.${NC}"
    exit 1
fi

# Function to handle cleanup on exit
cleanup() {
    echo -e "\n${BLUE}Shutting down servers...${NC}"
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

# Set up trap to handle Ctrl+C
trap cleanup SIGINT SIGTERM

# Start backend server
echo -e "${GREEN}Starting backend server...${NC}"
python3 server.py &
BACKEND_PID=$!

# Check if backend started successfully
sleep 2
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo -e "${RED}Failed to start backend server.${NC}"
    cleanup
fi

echo -e "${GREEN}Backend server running with PID: $BACKEND_PID${NC}"

# Start frontend server
echo -e "${GREEN}Starting frontend server...${NC}"
cd front && npm start &
FRONTEND_PID=$!

# Check if frontend started successfully
sleep 5
if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    echo -e "${RED}Failed to start frontend server.${NC}"
    cleanup
fi

echo -e "${GREEN}Frontend server running with PID: $FRONTEND_PID${NC}"
echo -e "${BLUE}-------------------------------------${NC}"
echo -e "${GREEN}QuantAI Hospital is now running!${NC}"
echo -e "${GREEN}Backend: http://localhost:8000${NC}"
echo -e "${GREEN}Frontend: http://localhost:3000${NC}"
echo -e "${BLUE}-------------------------------------${NC}"
echo -e "${BLUE}Press Ctrl+C to stop all servers${NC}"

# Wait for user to press Ctrl+C
wait 