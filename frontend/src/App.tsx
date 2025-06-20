import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Toaster, toast } from 'react-hot-toast';

import { ThemeProvider } from './contexts/ThemeContext';
import Header from './components/Header';
import Footer from './components/Footer';
import TextMode from './components/TextMode';
import VoiceMode from './components/VoiceMode';
import SplashScreen from './components/SplashScreen';
import Hero from './components/Hero';
import { Message, Mode } from './types';
import { buttonHover } from './utils/animations';

// Sample welcome message
const WELCOME_MESSAGE: Message = {
  id: '1',
  text: "G'day! I'm the QuantAI Hospital Assistant. How can I help you today?",
  isUser: false,
  timestamp: new Date(),
};

const App: React.FC = () => {
  const [mode, setMode] = useState<Mode>('text');
  const [messages, setMessages] = useState<Message[]>([WELCOME_MESSAGE]);
  const [isLoading, setIsLoading] = useState(false);
  const [isTransitioning, setIsTransitioning] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [showingSplash, setShowingSplash] = useState(true);
  const [chatStarted, setChatStarted] = useState(false);

  useEffect(() => {
    // Force dark mode
    document.documentElement.classList.add('dark');
    
    // Hide splash screen after it completes
    const timer = setTimeout(() => {
      setShowingSplash(false);
    }, 3000);
    
    return () => clearTimeout(timer);
  }, []);

  // Mock function for sending a message to the backend
  const sendMessageToBackend = async (text: string, language: string): Promise<string> => {
    // In a real implementation, this would call an API
    console.log('Sending message to backend:', text, 'lang:', language);

    try {
      const res = await fetch('http://localhost:8005/text-query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text, language })
      });
      const data = await res.json();
      if (data.success && data.response) {
        return data.response;
      }
    } catch (err) {
      console.error('API call failed, falling back to mock response', err);
    }
    
    // Simulate a delay for a more realistic experience
    await new Promise(resolve => setTimeout(resolve, 1500 + Math.random() * 1000));
    
    // Mock response with different types of responses
    const responses = [
      `I understand you're asking about "${text}". Here's what I can tell you based on our hospital procedures and guidelines.`,
      `Regarding "${text}", according to our standard operating procedures, here are the key points you should know.`,
      `For your question about "${text}", I can provide guidance based on our current protocols and best practices.`,
      `Let me help you with "${text}". This is covered in our hospital manual under the relevant section.`
    ];
    
    return responses[Math.floor(Math.random() * responses.length)];
  };

  // Handler for sending text messages
  const handleSendTextMessage = async (text: string, language: string) => {
    if (!chatStarted) {
      setChatStarted(true);
    }
    
    // Add user message to chat
    const userMessage: Message = {
      id: Date.now().toString(),
      text,
      isUser: true,
      timestamp: new Date(),
    };
    
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);
    
    try {
      // Get response from backend
      const response = await sendMessageToBackend(text, language);
      
      // Add bot response to chat
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: response,
        isUser: false,
        timestamp: new Date(),
      };
      
      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      
      // Show error toast
      toast.error('Failed to send message. Please try again.', {
        duration: 4000,
        position: 'top-right',
      });
    } finally {
      setIsLoading(false);
    }
  };

  // Handler for sending voice messages
  const handleSendVoiceMessage = async (audioBlob?: Blob, language: string = 'english') => {
    if (!chatStarted) {
      setChatStarted(true);
    }
    
    // Add user voice message indicator to chat
    const userMessage: Message = {
      id: Date.now().toString(),
      text: "[Voice message sent]",
      isUser: true,
      timestamp: new Date(),
    };
    
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);
    
    try {
      // Get response from backend (in a real implementation, we would send the audio blob)
      const response = await sendMessageToBackend("Voice message transcription would go here", language);
      
      // Add bot response to chat
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: response,
        isUser: false,
        timestamp: new Date(),
      };
      
      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error sending voice message:', error);
      
      // Show error toast
      toast.error('Failed to process voice message. Please try again.', {
        duration: 4000,
        position: 'top-right',
      });
    } finally {
      setIsLoading(false);
    }
  };

  // Handle switching between Text and Voice modes with animation
  const handleModeChange = (newMode: Mode) => {
    if (mode === newMode) return;
    
    setIsTransitioning(true);
    setTimeout(() => {
      setMode(newMode);
      setIsTransitioning(false);
    }, 300);
  };

  // Handle settings toggle
  const handleSettingsToggle = () => {
    setShowSettings(!showSettings);
  };

  // Handle starting chat from hero section
  const handleStartChat = (initialText?: string) => {
    setChatStarted(true);
    if (initialText) {
      handleSendTextMessage(initialText, 'english');
    }
  };

  // If splash screen is showing, render only the splash screen
  if (showingSplash) {
    return <SplashScreen onComplete={() => setShowingSplash(false)} />;
  }

  const pageTransition = {
    hidden: { opacity: 0, y: 20 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: {
        duration: 0.6,
        ease: [0.6, -0.05, 0.01, 0.99]
      }
    },
    exit: { 
      opacity: 0, 
      y: -20,
      transition: {
        duration: 0.4
      }
    }
  };

  return (
    <ThemeProvider>
      <div className="flex flex-col h-screen bg-black text-white font-['DM_Sans']">
        {/* Toast Notifications */}
        <Toaster />
        
        {/* Header - Only show when chat is started */}
        <AnimatePresence>
          {chatStarted && (
            <motion.div
              initial={{ opacity: 0, y: -50 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -50 }}
              transition={{ duration: 0.4 }}
              className="z-10 relative"
            >
              <Header 
                
              />
            </motion.div>
          )}
        </AnimatePresence>
        
        {/* Mode Selector - Only show when chat is started */}
        <AnimatePresence>
          {chatStarted && (
            <motion.div 
              className="font-['DM_Sans'] container mx-auto px-4 py-3 flex justify-center z-10 relative"
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ delay: 0.1, duration: 0.4 }}
            >
              <div className="bg-black/60 rounded-2xl p-1 inline-flex shadow-xl border border-gray-800">
                <motion.button
                  onClick={() => handleModeChange('text')}
                  variants={buttonHover}
                  whileHover="hover"
                  whileTap="tap"
                  className={`px-6 py-3 rounded-xl transition-all duration-300 font-medium ${
                    mode === 'text'
                      ? 'bg-police-blue text-white shadow-lg'
                      : 'text-gray-300 hover:bg-gray-800'
                  }`}
                >
                  Text Mode
                </motion.button>
                <motion.button
                  onClick={() => handleModeChange('voice')}
                  variants={buttonHover}
                  whileHover="hover"
                  whileTap="tap"
                  className={`px-6 py-3 rounded-xl transition-all duration-300 font-medium ${
                    mode === 'voice'
                      ? 'bg-police-blue text-white shadow-lg'
                      : 'text-gray-300 hover:bg-gray-800'
                  }`}
                >
                  Voice Mode
                </motion.button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
        
        {/* Main Content */}
        <main className="flex-1 overflow-hidden container mx-auto px-4 py-3 mb-auto relative">
          <AnimatePresence mode="wait">
            {!chatStarted ? (
              <div className="h-full w-full absolute inset-0">
                <Hero 
                  onStartChat={handleStartChat} 
                  onModeChange={handleModeChange} 
                />
              </div>
            ) : (
              <motion.div
                key="chat"
                initial="hidden"
                animate="visible"
                exit="exit"
                variants={pageTransition}
                className={`rounded-2xl h-full overflow-hidden ${isTransitioning ? 'opacity-0' : 'opacity-100'}`}
              >
                {mode === 'text' ? (
                  <TextMode
                    messages={messages}
                    onSendMessage={handleSendTextMessage}
                    isLoading={isLoading}
                  />
                ) : (
                  <VoiceMode
                    messages={messages}
                    setMessages={setMessages}
                  />
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </main>
        
        {/* Footer */}
        <Footer />
        
      </div>
    </ThemeProvider>
  );
};

export default App; 