# QuantAI Hospital Assistant

A sophisticated AI healthcare assistant with both text and voice interaction capabilities.

## Features

- Interactive chat interface with AI healthcare assistant
- Voice input and output capabilities
- Multiple language support
- Responsive design for all device sizes
- 3D parallax effects and modern UI
- Sidebar with dummy functionality for demonstration

## Prerequisites

- Python 3.8+
- Node.js 16+
- npm 8+

## Installation

1. Clone this repository
2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Install frontend dependencies:

```bash
cd front
npm install
cd ..
```

## Running the Application

### Option 1: Using the start script (recommended)

The easiest way to run both the backend and frontend is using the start script:

```bash
./start.sh
```

This will start both the backend server on port 8000 and the frontend server on port 3000.

### Option 2: Running servers separately

To run the backend server:

```bash
python3 server.py
```

To run the frontend server:

```bash
cd front
npm start
```

## Troubleshooting

### API Connection Issues

If the chat screen goes blank or doesn't display responses:

1. Make sure both backend and frontend servers are running
2. Check the browser console for errors
3. Verify that the proxy is correctly configured in `front/vite.config.ts`
4. Try using the "Test" button (visible in development mode) to add a mock response

### React Hooks Error

If you see an error about "rendered more hooks than during the previous render":

1. This is likely due to hooks being called conditionally or in loops
2. Check the `getMessageTransform` function in `ChatScreen.tsx` to ensure hooks are called consistently

### Backend Server Issues

If the backend server fails to start:

1. Check that all required Python packages are installed
2. Verify that port 8000 is not in use by another application
3. Check the server logs for specific errors

## Project Structure

- `server.py` - FastAPI backend server
- `agent.py` - Text processing agent
- `voice_agent.py` - Voice processing agent
- `front/` - React frontend application
  - `src/App.tsx` - Main application component
  - `src/components/ChatScreen/` - Chat interface component

## License

This project is a proof-of-concept and is not intended for production use.

## Disclaimer

This application is for demonstration purposes only. Any medical information provided should not replace professional medical advice. 