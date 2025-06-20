import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  InformationCircleIcon, 
  Cog6ToothIcon
} from '@heroicons/react/24/outline';
import { buttonHover } from '../utils/animations';


const Header: React.FC = () => {
  const [showInfo, setShowInfo] = useState(false);

  return (
    <motion.header 
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="bg-black border-b border-gray-800 shadow-lg relative z-10"
    >
      <div className="container mx-auto px-4 py-4 flex items-center justify-between">
        <motion.div 
          className="flex items-center gap-4"
          whileHover={{ scale: 1.02 }}
          transition={{ duration: 0.2 }}
        >
          <motion.img 
            src="/logoQN.png" 
            alt="QuantAI NZ Logo" 
            className="h-12 rounded-xl"
            whileHover={{ rotate: 5 }}
            transition={{ duration: 0.3 }}
          />
          <motion.h1 
            className="text-white text-xl md:text-2xl font-bold"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
          >
            QuantAI Hospital Assistant
          </motion.h1>
        </motion.div>

        <div className="flex items-center gap-3">
         

          {/* Info Button */}
          <motion.button
            variants={buttonHover}
            whileHover="hover"
            whileTap="tap"
            onClick={() => setShowInfo(!showInfo)}
            className="relative flex items-center gap-2 backdrop-blur-md bg-[#1a1a1a]/80 rounded-xl px-3 py-2 text-white hover:bg-[#2a2a2a]/80 transition-colors"
            aria-label="Show information about this application"
          >
            <InformationCircleIcon className="h-5 w-5" />
            <span className="hidden sm:inline">Info</span>

            {/* Info Tooltip */}
            <AnimatePresence>
              {showInfo && (
                <motion.div
                  initial={{ opacity: 0, y: 10, scale: 0.9 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: 10, scale: 0.9 }}
                  className="absolute top-full right-0 mt-2 w-80 backdrop-blur-xl bg-[#111]/90 border border-gray-800 rounded-2xl p-4 z-50 shadow-2xl"
                >
                  <div className="text-sm text-gray-300 space-y-2">
                    <h4 className="font-semibold text-police-blue">About This Assistant</h4>
                    <p>This AI assistant helps with:</p>
                    <ul className="list-disc list-inside space-y-1 text-gray-400">
                      <li>Quick access to hospital procedures</li>
                      <li>Emergency response guidance</li>
                      <li>Report generation assistance</li>
                    </ul>
                    <div className="mt-3 p-2 bg-yellow-900/20 rounded-xl border-l-4 border-yellow-600">
                      <p className="text-xs text-yellow-200">
                        <strong>Note:</strong> This is a Proof of Concept by QuantAI, NZ. 
                        Not intended for production use.
                      </p>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.button>

          <div className="hidden md:block">
            <motion.span 
              className="rounded-xl bg-[#1a1a1a] text-white text-xs font-bold px-3 py-1"
              whileHover={{ scale: 1.1 }}
              transition={{ duration: 0.2 }}
            >
              PoC
            </motion.span>
          </div>
        </div>
      </div>
    </motion.header>
  );
};

export default Header; 