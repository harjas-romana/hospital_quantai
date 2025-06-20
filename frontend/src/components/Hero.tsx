import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FaMicrophone, FaKeyboard, FaPaperPlane, FaShieldAlt, FaHospital, FaHospitalAlt } from 'react-icons/fa';

interface HeroProps {
  onStartChat: (text?: string) => void;
  onModeChange: (mode: 'text' | 'voice') => void;
}

const Hero: React.FC<HeroProps> = ({ onStartChat, onModeChange }) => {
  const [inputText, setInputText] = useState('');
  const [mode, setMode] = useState<'text' | 'voice'>('text');
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    
    // Preload background gif for smoother experience
    const img = new Image();
    img.src = '/back.gif';
    
    return () => setMounted(false);
  }, []);

  const handleModeToggle = (newMode: 'text' | 'voice') => {
    setMode(newMode);
    onModeChange(newMode);
  };

  const handleStartChat = () => {
    if (inputText.trim()) {
      onStartChat(inputText);
    } else {
      onStartChat();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleStartChat();
    }
  };

  // Animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: { 
      opacity: 1,
      transition: { 
        when: "beforeChildren",
        staggerChildren: 0.2,
        duration: 0.8,
        ease: [0.6, -0.05, 0.01, 0.99]
      }
    },
    exit: {
      opacity: 0,
      transition: {
        duration: 0.5,
        ease: "easeInOut"
      }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: { 
      y: 0, 
      opacity: 1,
      transition: { duration: 0.6, ease: [0.6, -0.05, 0.01, 0.99] }
    }
  };

  const buttonVariants = {
    idle: { scale: 1 },
    hover: { 
      scale: 1.05,
      boxShadow: "0px 8px 15px rgba(0, 0, 0, 0.3)",
      transition: { duration: 0.3, ease: "easeOut" }
    },
    tap: { 
      scale: 0.95,
      boxShadow: "0px 2px 8px rgba(0, 0, 0, 0.2)",
      transition: { duration: 0.1 }
    }
  };

  const floatVariants = {
    animate: {
      y: [0, -10, 0],
      transition: {
        duration: 3,
        repeat: Infinity,
        repeatType: "reverse" as const,
        ease: "easeInOut"
      }
    }
  };

  return (
    <motion.div 
      className="font-['DM_Sans'] relative w-full h-full min-h-screen flex flex-col items-center justify-center overflow-hidden"
      initial="hidden"
      animate={mounted ? "visible" : "hidden"}
      exit="exit"
      variants={containerVariants}
    >
      {/* Fixed fullscreen background */}
      <div className="fixed inset-0 z-0 w-full h-full">
        <div 
          className="absolute inset-0 w-full h-full bg-cover bg-center bg-no-repeat"
          style={{ 
            backgroundImage: 'url(/back.gif)',
            filter: 'brightness(0.8) contrast(1.2) saturate(1.2)'
          }}
        />
        <div 
          className="absolute inset-0 bg-gradient-to-b from-black via-transparent to-black opacity-80"
        />
        <div className="absolute inset-0 bg-[#000011] opacity-70" />
      </div>

      {/* Animated particles or elements */}
      <div className="fixed inset-0 z-0 overflow-hidden">
        {[...Array(20)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 md:w-2 md:h-2 bg-police-blue rounded-full opacity-30"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
            }}
            animate={{
              opacity: [0.2, 0.5, 0.2],
              scale: [1, 1.5, 1],
            }}
            transition={{
              duration: 3 + Math.random() * 5,
              repeat: Infinity,
              delay: Math.random() * 2,
            }}
          />
        ))}
      </div>

      {/* Main Content */}
      <motion.div 
        className="font-['DM_Sans'] z-10 max-w-4xl w-full px-4 py-16 relative"
        variants={itemVariants}
      >
        <motion.div 
          className="absolute top-0 left-1/2 transform -translate-x-1/2 -translate-y-12 opacity-80"
          variants={floatVariants}
          animate="animate"
        >
          <motion.div 
            className="w-20 h-20 md:w-24 md:h-24 bg-police-blue rounded-full flex items-center justify-center shadow-lg"
            animate={{ 
              boxShadow: [
                '0 0 0 0 rgba(26, 59, 92, 0)',
                '0 0 0 20px rgba(26, 59, 92, 0.2)',
                '0 0 0 0 rgba(26, 59, 92, 0)'
              ]
            }}
            transition={{ 
              duration: 2,
              repeat: Infinity,
              repeatType: 'loop'
            }}
          >
            <FaHospitalAlt className="w-10 h-10 md:w-12 md:h-12 text-white" />
          </motion.div>
        </motion.div>

        <motion.h1 
          className="font-['DM_Sans'] text-4xl md:text-5xl lg:text-6xl font-bold mb-6 text-center text-white"
          variants={itemVariants}
        >
          <motion.span 
            className="inline-block"
            animate={{ 
              textShadow: [
                '0 0 10px rgba(255,255,255,0.1)',
                '0 0 20px rgba(255,255,255,0.2)',
                '0 0 10px rgba(255,255,255,0.1)'
              ]
            }}
            transition={{ duration: 3, repeat: Infinity }}
          >
            QuantAI's Hospital
          </motion.span>
          <br/>
          <motion.span 
            className="text-3xl md:text-4xl text-police-blue inline-block"
            animate={{ 
              color: ['#1a3b5c', '#2c5282', '#1a3b5c']
            }}
            transition={{ duration: 4, repeat: Infinity }}
          >
            AI Assistant
          </motion.span>
        </motion.h1>

        <motion.p 
          className="text-lg md:text-xl text-center mb-12 text-gray-300"
          variants={itemVariants}
        >
          Get instant answers to questions about hospital procedures, resources, and more
        </motion.p>

        {/* Main input container */}
        <motion.div 
          className="backdrop-blur-xl bg-black bg-opacity-30 border border-gray-800 rounded-3xl p-2 md:p-3 shadow-2xl max-w-3xl mx-auto"
          variants={itemVariants}
          whileHover={{ 
            boxShadow: '0 20px 40px rgba(0, 0, 0, 0.4), 0 0 0 1px rgba(255, 255, 255, 0.1) inset',
            transition: { duration: 0.3 }
          }}
        >
          <div className="flex items-center">
            {/* Mode toggles */}
            <div className="hidden md:flex items-center bg-black bg-opacity-60 rounded-full p-1.5 mr-3 border border-gray-800">
              <motion.button
                onClick={() => handleModeToggle('text')}
                className={`rounded-full p-2.5 ${mode === 'text' 
                  ? 'bg-police-blue text-white' 
                  : 'text-gray-400 hover:text-white'}`}
                variants={buttonVariants}
                whileHover="hover"
                whileTap="tap"
                aria-label="Switch to text mode"
              >
                <FaKeyboard className="h-5 w-5" />
              </motion.button>
              <motion.button
                onClick={() => handleModeToggle('voice')}
                className={`rounded-full p-2.5 ${mode === 'voice' 
                  ? 'bg-police-blue text-white' 
                  : 'text-gray-400 hover:text-white'}`}
                variants={buttonVariants}
                whileHover="hover"
                whileTap="tap"
                aria-label="Switch to voice mode"
              >
                <FaMicrophone className="h-5 w-5" />
              </motion.button>
            </div>

            <AnimatePresence mode="wait">
              {mode === 'text' ? (
                <motion.div 
                  key="text-mode"
                  className="flex-1 flex items-center"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  transition={{ duration: 0.3 }}
                >
                  <input
                    type="text"
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Type your question..."
                    className="flex-1 bg-transparent border-none focus:ring-0 text-lg md:text-xl py-4 px-4 text-white placeholder-gray-400"
                    autoFocus
                  />
                  <motion.button
                    variants={buttonVariants}
                    whileHover="hover"
                    whileTap="tap"
                    onClick={handleStartChat}
                    className="bg-police-blue text-white rounded-full p-3.5 ml-1 flex items-center justify-center"
                    aria-label="Send message"
                  >
                    <FaPaperPlane className="h-5 w-5" />
                  </motion.button>
                </motion.div>
              ) : (
                <motion.button
                  key="voice-mode"
                  onClick={handleStartChat}
                  className="flex-1 bg-transparent border-none focus:ring-0 text-lg md:text-xl py-5 px-4 flex items-center justify-center gap-3"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ duration: 0.3 }}
                >
                  <motion.div
                    animate={{ 
                      scale: [1, 1.2, 1],
                      boxShadow: [
                        '0 0 0 0 rgba(26, 59, 92, 0)',
                        '0 0 0 10px rgba(26, 59, 92, 0.3)',
                        '0 0 0 0 rgba(26, 59, 92, 0)'
                      ]
                    }}
                    transition={{ 
                      duration: 2,
                      repeat: Infinity,
                      repeatType: 'loop'
                    }}
                  >
                    <FaMicrophone className="h-8 w-8 text-police-blue" />
                  </motion.div>
                  <span className="text-gray-300">Tap to speak</span>
                </motion.button>
              )}
            </AnimatePresence>
          </div>
        </motion.div>
        
        {/* Example questions */}
        <motion.div 
          className="mt-12 grid grid-cols-1 md:grid-cols-2 gap-4 max-w-3xl mx-auto"
          variants={itemVariants}
        >
          {[
            "When is Doctor available for a consultation?",
            "What is the procedure for a patient admission?",
            "Help me draft an incident report for a patient fall.",
            "What are the guidelines for administering medication?"
          ].map((question, idx) => (
            <motion.button
              key={idx}
              variants={buttonVariants}
              whileHover="hover"
              whileTap="tap"
              onClick={() => {
                setInputText(question);
                onStartChat(question);
              }}
              className="backdrop-blur-xl bg-black bg-opacity-30 border border-gray-800 text-sm md:text-base text-left p-4 rounded-xl transition-all duration-200 text-gray-300 hover:text-white"
            >
              {question}
            </motion.button>
          ))}
        </motion.div>
      </motion.div>

      {/* Footer */}
      <motion.div 
        className="absolute bottom-6 md:bottom-10 text-center z-10"
        variants={itemVariants}
      >
        <p className="text-sm text-gray-500">
          Secured with end-to-end encryption | For official use only
        </p>
      </motion.div>
    </motion.div>
  );
};

export default Hero; 