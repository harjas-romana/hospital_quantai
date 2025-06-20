import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

interface SplashScreenProps {
  onComplete: () => void;
}

const SplashScreen: React.FC<SplashScreenProps> = ({ onComplete }) => {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const timer = setTimeout(() => onComplete(), 3000);
    
    // Simulate loading progress
    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          return 100;
        }
        return prev + 5;
      });
    }, 150);

    return () => {
      clearTimeout(timer);
      clearInterval(interval);
    };
  }, [onComplete]);

  return (
    <motion.div
      className="font-['DM_Sans'] fixed inset-0 z-50 flex flex-col items-center justify-center bg-light-bg dark:bg-dark-bg"
      initial={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.6, ease: "easeInOut" }}
    >
      <motion.div
        initial={{ scale: 0.8, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
        className="flex flex-col items-center"
      >
        <motion.img 
          src="/logoQN.png" 
          alt="Logo" 
          className="w-32 h-32 mb-8"
          animate={{ 
            rotateY: [0, 360],
          }}
          transition={{ 
            duration: 2, 
            repeat: Infinity, 
            ease: "easeInOut" 
          }}
        />
        
        <h1 className="text-3xl font-bold mb-6 text-police-blue">
          QuantAI's Hospital Assistant
        </h1>
        
        <div className="w-72 h-2 bg-gray-200 rounded-full overflow-hidden">
          <motion.div 
            className="h-full bg-police-blue"
            initial={{ width: '0%' }}
            animate={{ width: `${progress}%` }}
            transition={{ ease: "easeInOut" }}
          />
        </div>
        
        <p className="mt-4 text-gray-600 dark:text-gray-300">
          Loading resources...
        </p>
      </motion.div>
      
      <motion.div 
        className="absolute bottom-8"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.5, duration: 1 }}
      >
        <p className="text-sm text-gray-500 dark:text-gray-400">
          © {new Date().getFullYear()} QuantAI, NZ
        </p>
      </motion.div>
    </motion.div>
  );
};

export default SplashScreen; 