import React from 'react';
import { motion } from 'framer-motion';
import { fadeInUp } from '../utils/animations';

const Footer: React.FC = () => {
  return (
    <motion.footer 
      initial="initial"
      animate="animate"
      variants={fadeInUp}
      className="glass bg-gray-100/80 dark:bg-gray-900/80 border-t border-gray-200/50 dark:border-gray-700/50"
    >
      <div className="container mx-auto px-4 py-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-center md:text-left">
          {/* Disclaimer */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="md:col-span-2"
          >
            <p className="text-sm text-gray-600 dark:text-gray-400">
              <span className="font-medium text-police-blue dark:text-police-gold">Disclaimer:</span> This is a Proof of Concept by QuantAI, NZ.
              Not intended for production use.
            </p>
          </motion.div>

          {/* Version Info */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="text-center md:text-right"
          >
            <p className="text-xs text-gray-500 dark:text-gray-500">
              Version 1.0.0 | Built with React & TypeScript
            </p>
          </motion.div>
        </div>

        {/* Additional Info */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="mt-3 pt-3 border-t border-gray-200/50 dark:border-gray-700/50"
        >
          <div className="flex flex-col md:flex-row justify-between items-center text-xs text-gray-500 dark:text-gray-500 space-y-2 md:space-y-0">
            <div className="flex items-center space-x-4">
              <span className="flex items-center gap-1">
                <span className="text-green-500">🔒</span>
                Secure Communication
              </span>
              <span className="flex items-center gap-1">
                <span className="text-blue-500">⚡</span>
                Real-time Processing
              </span>
              <span className="flex items-center gap-1">
                <span className="text-purple-500">🎯</span>
                AI-Powered Responses
              </span>
            </div>
            <div className="flex items-center space-x-2">
              <span>Made with ❤️ by QuantAI, NZ</span>
            </div>
          </div>
        </motion.div>
      </div>
    </motion.footer>
  );
};

export default Footer; 