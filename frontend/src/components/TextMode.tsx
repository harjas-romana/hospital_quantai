import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { PaperAirplaneIcon } from '@heroicons/react/24/outline';
import { Message } from '../types';

interface TextModeProps {
  messages: Message[];
  onSendMessage: (text: string, language: string) => void;
  isLoading?: boolean;
}

const TextMode: React.FC<TextModeProps> = ({ messages, onSendMessage, isLoading = false }) => {
  const [inputText, setInputText] = useState('');
  const [isFocused, setIsFocused] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [languages, setLanguages] = useState<{ code: string; name: string }[]>([]);
  const [selectedLanguage, setSelectedLanguage] = useState('english');

  const springConfig = {
    type: "spring" as const,
    stiffness: 300,
    damping: 30,
    mass: 0.8
  };
  const customEasing = [0.25, 0.46, 0.45, 0.94];

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Fetch available languages once component mounts
  useEffect(() => {
    const fetchLanguages = async () => {
      try {
        const res = await fetch("http://localhost:8005/languages-text");
        const data = await res.json();
        if (data.success && data.languages) {
          const langs = Object.entries(data.languages).map(([code, name]) => ({ code, name: String(name) }));
          setLanguages(langs);
        }
      } catch (err) {
        console.error("Failed to fetch languages", err);
      }
    };
    fetchLanguages();
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputText.trim() && !isLoading) {
      onSendMessage(inputText.trim(), selectedLanguage);
      setInputText('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };
  
  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit'
    });
  };

  const messageVariants = {
    initial: { opacity: 0, y: 40, scale: 0.92 },
    animate: { 
      opacity: 1, y: 0, scale: 1,
      transition: { duration: 0.5, ease: customEasing } 
    },
    exit: { 
      opacity: 0, y: -20, scale: 0.95,
      transition: { duration: 0.3, ease: customEasing } 
    }
  };
  
  const buttonVariants = {
    hover: { 
      scale: 1.05,
      filter: 'brightness(110%)',
      transition: { duration: 0.3, ease: customEasing } 
    },
    tap: { 
      scale: 0.95,
      transition: { duration: 0.1 } 
    },
  };

  return (
    <div className="flex flex-col h-full font-['DM_Sans'] bg-black overflow-hidden">
      {/* Language Selector */}
      <div className="flex justify-end p-4">
        <select
          value={selectedLanguage}
          onChange={(e) => setSelectedLanguage(e.target.value)}
          className="bg-gray-800 text-white py-2 px-3 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          {languages.length === 0 ? (
            <option value="english">English</option>
          ) : (
            languages.map((lang) => (
              <option key={lang.code} value={lang.name.toLowerCase()}>
                {lang.name}
              </option>
            ))
          )}
        </select>
      </div>
      {/* Messages Container */}
      <div 
        className="flex-1 overflow-y-auto px-4"
        style={{
            scrollBehavior: 'smooth',
            WebkitOverflowScrolling: 'touch',
            scrollbarWidth: 'thin',
            scrollbarColor: 'rgba(75, 85, 99, 0.3) transparent'
        }}
    >
        <div className="max-w-[800px] mx-auto py-6">
          <AnimatePresence initial={false}>
            {messages.map((message) => (
              <motion.div
                key={message.id}
                layout
                variants={messageVariants}
                initial="initial"
                animate="animate"
                exit="exit"
                className={`mb-5 flex ${message.isUser ? 'justify-end' : 'justify-start'}`}
              >
                <div className={`max-w-[75%] ${message.isUser ? 'order-last' : ''}`}>
                  <motion.div
                    whileHover={{ 
                      scale: 1.02,
                      boxShadow: message.isUser 
                        ? '0 8px 48px rgba(0, 122, 255, 0.18)' 
                        : '0 8px 48px rgba(0, 0, 0, 0.18)',
                      transition: { duration: 0.2 }
                    }}
                    className={`px-6 py-4 min-h-[48px] rounded-2xl relative
                      ${message.isUser
                        ? 'bg-[#007AFF]/20 border border-[#007AFF]/30 text-white rounded-br-sm'
                        : 'bg-[#1A1A1A]/40 border border-[#333333]/30 text-gray-100 rounded-bl-sm'
                      }`}
                    style={{
                      backdropFilter: 'blur(24px)',
                      WebkitBackdropFilter: 'blur(24px)',
                      boxShadow: message.isUser 
                        ? '0 8px 32px rgba(0, 122, 255, 0.15)' 
                        : '0 8px 32px rgba(0, 0, 0, 0.15)'
                    }}
                  >
                    <p className="text-[16px] leading-[24px] font-normal tracking-[0.01em]">{message.text}</p>
                    <div className="flex justify-between items-center mt-2">
                      <span className={`text-[12px] leading-[16px] font-medium ${
                        message.isUser ? 'text-blue-200/70' : 'text-gray-400'
                      }`}>{formatTime(message.timestamp)}</span>
                    </div>
                  </motion.div>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
          {isLoading && (
            <motion.div 
              className="flex justify-start mb-5"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <div className="bg-gray-800/20 border border-gray-700/30 px-6 py-4 rounded-2xl rounded-bl-sm"
                style={{ backdropFilter: 'blur(24px)', WebkitBackdropFilter: 'blur(24px)' }}
              >
                <div className="flex items-center space-x-2">
                  <motion.div className="w-2 h-2 rounded-full bg-gray-300" animate={{ y: [0, -4, 0] }} transition={{ duration: 0.8, repeat: Infinity, ease: 'easeInOut' }}></motion.div>
                  <motion.div className="w-2 h-2 rounded-full bg-gray-300" animate={{ y: [0, -4, 0] }} transition={{ duration: 0.8, delay: 0.1, repeat: Infinity, ease: 'easeInOut' }}></motion.div>
                  <motion.div className="w-2 h-2 rounded-full bg-gray-300" animate={{ y: [0, -4, 0] }} transition={{ duration: 0.8, delay: 0.2, repeat: Infinity, ease: 'easeInOut' }}></motion.div>
                </div>
              </div>
            </motion.div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Form */}
      <motion.div 
        className="z-10 p-4"
        initial={{ y: 50, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.1, ...springConfig }}
        style={{ flexShrink: 0 }}
      >
        <motion.form 
          onSubmit={handleSubmit}
          className="relative"
        >
          <div 
            className="flex items-center p-2 rounded-2xl"
            style={{
              background: 'rgba(26, 26, 26, 0.7)',
              backdropFilter: 'blur(24px)',
              WebkitBackdropFilter: 'blur(24px)',
              border: '1px solid rgba(255, 255, 255, 0.05)',
              boxShadow: isFocused ? '0 8px 32px rgba(0, 122, 255, 0.3)' : '0 8px 32px rgba(0, 0, 0, 0.2)'
            }}
          >
            <input
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={handleKeyDown}
              onFocus={() => setIsFocused(true)}
              onBlur={() => setIsFocused(false)}
              placeholder="Type your message here..."
              className="flex-1 px-4 py-3 bg-transparent text-white placeholder-gray-400 focus:outline-none text-[16px]"
              disabled={isLoading}
            />
            
            <motion.button
              type="submit"
              disabled={isLoading || !inputText.trim()}
              variants={buttonVariants}
              whileHover={!isLoading && inputText.trim() ? "hover" : {}}
              whileTap={!isLoading && inputText.trim() ? "tap" : {}}
              className="p-3 rounded-xl focus:outline-none"
              style={{
                background: !isLoading && inputText.trim()
                  ? 'linear-gradient(135deg, #007AFF, #0056B3)'
                  : 'rgba(50, 50, 50, 0.8)',
                color: !isLoading && inputText.trim() ? 'white' : '#6b7280',
                cursor: !isLoading && inputText.trim() ? 'pointer' : 'not-allowed'
              }}
              aria-label="Send message"
            >
              <PaperAirplaneIcon className="h-6 w-6" />
            </motion.button>
          </div>
        </motion.form>
      </motion.div>
    </div>
  );
};

export default TextMode;