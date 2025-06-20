import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  MicrophoneIcon, 
  StopIcon, 
  SpeakerWaveIcon,
  SpeakerXMarkIcon,
  ArrowPathIcon
} from '@heroicons/react/24/outline';
import { Message } from '../types';

interface Recording {
  isRecording: boolean;
}

interface VoiceModeProps {
  messages: Message[];
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
}

type SpeechRecognitionType = any;

const VoiceMode: React.FC<VoiceModeProps> = ({ messages, setMessages }) => {
  const [recording, setRecording] = useState<Recording>({
    isRecording: false,
  });
  const [isListening, setIsListening] = useState(false);
  const [volume, setVolume] = useState(75);
  const [isMuted, setIsMuted] = useState(false);
  const [speakingAnimation, setSpeakingAnimation] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [languages, setLanguages] = useState<{ code: string; name: string }[]>([]);
  const [selectedLanguage, setSelectedLanguage] = useState('english');
  const [isLoading, setIsLoading] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const mediaChunks = useRef<Blob[]>([]);
  const recognitionRef = useRef<SpeechRecognitionType | null>(null);
  const draftMessageId = useRef<string | null>(null);
  const [pendingAudio, setPendingAudio] = useState<string | null>(null);

  // Spring physics for animations
  const springConfig = {
    type: "spring" as const,
    stiffness: 300,
    damping: 30,
    mass: 0.8
  };

  // Custom easing curves
  const customEasing = [0.25, 0.46, 0.45, 0.94]; // "easeOutQuart"
  const bouncyEasing = [0.68, -0.55, 0.265, 1.55]; // "easeOutBack"

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    
    // Simulate AI speaking when a new AI message arrives
    if (messages.length > 0 && !messages[messages.length - 1].isUser) {
      setSpeakingAnimation(true);
      const timer = setTimeout(() => {
        setSpeakingAnimation(false);
      }, 4000);
      return () => clearTimeout(timer);
    }
  }, [messages]);

  useEffect(() => {
    const fetchLangs = async () => {
      try {
        const res = await fetch("http://localhost:8005/languages-voice");
        const data = await res.json();
        if (data.success && data.languages) {
          const langs = Object.entries(data.languages).map(([code, name]) => ({ code, name: String(name) }));
          setLanguages(langs);
        }
      } catch (err) {
        console.error("Failed to fetch voice languages", err);
      }
    };
    fetchLangs();
  }, []);

  const appendMessage = (msg: Message) => {
    setMessages(prev => [...prev, msg]);
  };

  const updateDraftMessage = (text: string) => {
    setMessages(prev => prev.map(m => (m.id === draftMessageId.current ? { ...m, text } : m)));
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      mediaChunks.current = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) mediaChunks.current.push(e.data);
      };
      recorder.start();
      setMediaRecorder(recorder);
      setRecording({ isRecording: true });

      // start speech recognition
      const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
      if (SpeechRecognition) {
        const recognition = new SpeechRecognition();
        recognition.interimResults = true;
        recognition.continuous = true;
        recognition.lang = selectedLanguage === 'english' ? 'en-US' : undefined;
        recognition.onresult = (event: any) => {
          let interim = '';
          let final = '';
          for (let i = event.resultIndex; i < event.results.length; ++i) {
            const transcript = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
              final += transcript;
            } else {
              interim += transcript;
            }
          }
          const combined = (final || interim).trim();
          if (combined) {
            if (!draftMessageId.current) {
              const id = Date.now().toString();
              draftMessageId.current = id;
              appendMessage({ id, text: combined, isUser: true, timestamp: new Date() });
            } else {
              updateDraftMessage(combined + (interim ? ' …' : ''));
            }
          }
        };
        recognitionRef.current = recognition;
        recognition.start();
      }
    } catch (err) {
      console.error('Microphone permission denied or error:', err);
    }
  };

  const stopRecording = () => {
    recognitionRef.current?.stop();
    setIsListening(false);
    setRecording({ isRecording: false });
    mediaRecorder?.stop();
  };

  const toggleRecording = () => {
    if (isLoading) return;
    if (recording.isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  useEffect(() => {
    if (!mediaRecorder) return;
    const handleStop = async () => {
      setIsLoading(true);
      const blob = new Blob(mediaChunks.current, { type: 'audio/webm' });
      const form = new FormData();
      form.append('audio_file', blob, 'recording.webm');
      form.append('language', selectedLanguage);
      try {
        const res = await fetch('http://localhost:8005/voice-query', {
          method: 'POST',
          body: form,
        });
        const data = await res.json();
        if (res.ok && data) {
          // Update user message with server-transcribed text if different
          if (data.user_text && draftMessageId.current) {
            updateDraftMessage(data.user_text);
          }
          // Append assistant response
          appendMessage({ id: Date.now().toString(), text: data.response_text, isUser: false, timestamp: new Date() });
          // Auto-play TTS audio
          if (data.audio_url) {
            playAudio(`http://localhost:8005${data.audio_url}`);
          }
        } else {
          console.error('Voice query failed', data);
        }
      } catch (e) {
        console.error('Voice query error', e);
      } finally {
        setIsLoading(false);
        draftMessageId.current = null;
      }
    };
    mediaRecorder.addEventListener('stop', handleStop);
    return () => {
      mediaRecorder.removeEventListener('stop', handleStop);
    };
  }, [mediaRecorder]);

  // Format timestamp
  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit'
    });
  };

  // Message animation variants
  const messageVariants = {
    initial: { 
      opacity: 0, 
      y: 40, 
      scale: 0.92 
    },
    animate: { 
      opacity: 1, 
      y: 0, 
      scale: 1,
      transition: { 
        duration: 0.5, 
        ease: customEasing
      } 
    },
    exit: { 
      opacity: 0, 
      y: -20, 
      scale: 0.95,
      transition: { 
        duration: 0.3, 
        ease: customEasing
      } 
    }
  };

  // Waveform animation variants
  const waveformVariants = {
    listening: (i: number) => ({
      height: recording.isRecording ? [4, 30 + Math.random() * 20, 4] : 4,
      backgroundColor: recording.isRecording ? 
        ['#007AFF', '#0096FF', '#007AFF'] : 
        ['#374151', '#374151', '#374151'],
      transition: { 
        duration: 0.6 + Math.random() * 0.4, 
        repeat: Infinity, 
        delay: i * 0.04,
        ease: "easeInOut"
      }
    }),
    idle: { height: 4, backgroundColor: '#374151' }
  };

  // Speaking animation variants
  const speakingVariants = {
    speaking: (i: number) => ({
      height: speakingAnimation ? [4, 20 + Math.random() * 15, 4] : 4,
      backgroundColor: speakingAnimation ? 
        ['#00D084', '#34d399', '#00D084'] : 
        ['#374151', '#374151', '#374151'],
      transition: { 
        duration: 0.8 + Math.random() * 0.4, 
        repeat: Infinity, 
        delay: i * 0.03,
        ease: "easeInOut"
      }
    }),
    idle: { height: 4, backgroundColor: '#374151' }
  };

  // Button animation variants
  const buttonVariants = {
    idle: { scale: 1 },
    hover: { 
      scale: 1.02, 
      filter: 'brightness(110%)',
      boxShadow: '0 8px 32px rgba(0, 122, 255, 0.3)',
      transition: { duration: 0.3, ease: customEasing } 
    },
    tap: { 
      scale: 0.96, 
      filter: 'brightness(90%)',
      boxShadow: '0 2px 8px rgba(0, 122, 255, 0.2)',
      transition: { duration: 0.1 } 
    },
    disabled: { opacity: 0.5, scale: 1 }
  };

  // Mic button pulse animation
  const pulseVariants = {
    recording: {
      scale: [1, 1.15, 1],
      boxShadow: [
        '0 0 0 0 rgba(0, 122, 255, 0)',
        '0 0 0 20px rgba(0, 122, 255, 0.3)',
        '0 0 0 0 rgba(0, 122, 255, 0)'
      ],
      transition: {
        duration: 2,
        repeat: Infinity,
        ease: "easeInOut"
      }
    },
    idle: {
      scale: 1,
      boxShadow: '0 8px 32px rgba(0, 0, 0, 0.2)'
    }
  };

  const playAudio = (url: string) => {
    try {
      const audio = new Audio(url);
      audio.crossOrigin = 'anonymous';
      const playPromise = audio.play();
      if (playPromise !== undefined) {
        playPromise.catch(() => {
          setPendingAudio(url);
        });
      }
    } catch (err) {
      console.error('Auto-play failed', err);
      setPendingAudio(url);
    }
  };

  return (
    <div className="flex flex-col h-full font-['DM_Sans'] overflow-hidden bg-black">
      {/* Header Section */}
      <motion.div 
        className="z-20 h-20"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={springConfig}
        style={{
          background: 'rgba(10, 10, 10, 0.7)',
          backdropFilter: 'blur(24px)',
          WebkitBackdropFilter: 'blur(24px)',
          borderBottom: '1px solid rgba(255, 255, 255, 0.05)',
          flexShrink: 0,
        }}
      >
        <div className="max-w-[800px] mx-auto px-8 h-full flex items-center justify-between">
          <motion.h1 
            className="text-[20px] font-semibold text-white leading-[24px] tracking-[-0.02em]"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1, ...springConfig }}
          >
            Voice Assistant
          </motion.h1>
          
          {/* Language selector */}
          <select
            value={selectedLanguage}
            onChange={(e) => setSelectedLanguage(e.target.value)}
            className="bg-gray-800 text-white py-2 px-3 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {languages.length === 0 ? (
              <option value="english">English</option>
            ) : (
              languages.map((lang) => (
                <option key={lang.code} value={lang.name.toLowerCase()}>{lang.name}</option>
              ))
            )}
          </select>
          
          <motion.div
            animate={{ 
              opacity: isLoading || recording.isRecording || speakingAnimation ? [0.8, 1, 0.8] : 1 
            }}
            transition={{ duration: 1.5, repeat: isLoading || recording.isRecording || speakingAnimation ? Infinity : 0 }}
            className={`flex items-center px-4 py-2 rounded-full transition-all duration-300
              ${isLoading ? 'bg-[#FFB800]/10 border border-[#FFB800]/20 text-[#FFB800]' : 
                recording.isRecording ? 'bg-[#007AFF]/10 border border-[#007AFF]/20 text-[#007AFF]' : 
                speakingAnimation ? 'bg-[#00D084]/10 border border-[#00D084]/20 text-[#00D084]' :
                'bg-[#1A1A1A]/60 border border-[#333333]/40 text-gray-400'}`}
            style={{
              backdropFilter: 'blur(16px)',
              WebkitBackdropFilter: 'blur(16px)',
              boxShadow: '0 4px 24px rgba(0, 0, 0, 0.12)'
            }}
          >
            <div className={`w-2 h-2 rounded-full mr-2 transition-all duration-300
              ${isLoading ? 'bg-[#FFB800] animate-pulse' : 
                recording.isRecording ? 'bg-[#007AFF] animate-pulse' : 
                speakingAnimation ? 'bg-[#00D084] animate-pulse' :
                'bg-gray-500'}`} 
            />
            <span className="text-[12px] font-medium leading-[16px]">
              {isLoading ? "Processing..." : 
                recording.isRecording ? "Listening..." : 
                speakingAnimation ? "Speaking..." :
                "Ready"}
            </span>
          </motion.div>
        </div>
      </motion.div>

      {/* Chat Container */}
      <motion.div 
        className="px-4 overflow-y-auto flex-1"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
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
                className={`flex mb-5 ${message.isUser ? 'justify-end' : 'justify-start'}`}
              >
                <div className={`max-w-[65%] ${message.isUser ? 'order-last' : ''}`}>
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
                    <div className="flex items-start mb-2">
                      {!message.isUser && speakingAnimation && message === messages[messages.length - 1] && (
                        <div className="flex items-end space-x-1 h-4 mr-3 mt-1">
                          {[0, 1, 2, 3, 4].map((i) => (
                            <motion.div
                              key={i}
                              className="w-[2px] rounded-full bg-[#00D084]"
                              custom={i}
                              variants={speakingVariants}
                              animate="speaking"
                            />
                          ))}
                        </div>
                      )}
                      <p className="text-[16px] leading-[24px] font-normal tracking-[0.01em]">{message.text}</p>
                    </div>
                    
                    <div className="flex justify-between items-center mt-2">
                      <span className={`text-[12px] leading-[16px] font-medium ${
                        message.isUser ? 'text-blue-200/70' : 'text-gray-400'
                      }`}>{formatTime(message.timestamp)}</span>
                      
                      {message.isUser && (
                        <motion.div
                          animate={{ opacity: [0.7, 1, 0.7] }}
                          transition={{ duration: 2, repeat: Infinity }}
                          className="text-[12px] text-blue-200/70 ml-3 font-mono"
                        >
                          ✓✓
                        </motion.div>
                      )}
                    </div>
                  </motion.div>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
          <div ref={messagesEndRef} />
        </div>
      </motion.div>

      {/* Voice Control Panel */}
      <motion.div 
        className="z-10 h-[320px]"
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2, ...springConfig }}
        style={{
          background: 'rgba(10, 10, 10, 0.8)',
          backdropFilter: 'blur(32px)',
          WebkitBackdropFilter: 'blur(32px)',
          borderTop: '1px solid rgba(255, 255, 255, 0.05)',
          boxShadow: '0 -16px 64px rgba(0, 0, 0, 0.24)',
          flexShrink: 0,
        }}
      >
        <div className="max-w-[800px] mx-auto px-8 py-6 h-full">
          <div className="flex flex-col h-full">
            {/* Main Recording Button - 140px diameter */}
            <div className="flex justify-center mb-8">
              <motion.div
                className="relative"
                variants={buttonVariants}
                initial="idle"
                whileHover={isLoading ? "disabled" : "hover"}
                whileTap={isLoading ? "disabled" : "tap"}
              >
                {/* Triple-ring pulse animation */}
                <AnimatePresence>
                  {recording.isRecording && (
                    <>
                      <motion.div
                        initial={{ scale: 0.8, opacity: 0 }}
                        animate={{ 
                          scale: [0.8, 1.5, 0.8], 
                          opacity: [0, 0.3, 0] 
                        }}
                        exit={{ scale: 0.8, opacity: 0 }}
                        className="absolute inset-0 rounded-full"
                        style={{ backgroundColor: 'rgba(0, 122, 255, 0.4)' }}
                        transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
                      />
                      <motion.div
                        initial={{ scale: 0.6, opacity: 0 }}
                        animate={{ 
                          scale: [0.6, 1.8, 0.6], 
                          opacity: [0, 0.2, 0] 
                        }}
                        exit={{ scale: 0.6, opacity: 0 }}
                        className="absolute inset-0 rounded-full"
                        style={{ backgroundColor: 'rgba(0, 122, 255, 0.2)' }}
                        transition={{ duration: 2.5, repeat: Infinity, delay: 0.3, ease: "easeInOut" }}
                      />
                      <motion.div
                        initial={{ scale: 0.4, opacity: 0 }}
                        animate={{ 
                          scale: [0.4, 2.1, 0.4], 
                          opacity: [0, 0.1, 0] 
                        }}
                        exit={{ scale: 0.4, opacity: 0 }}
                        className="absolute inset-0 rounded-full"
                        style={{ backgroundColor: 'rgba(0, 122, 255, 0.1)' }}
                        transition={{ duration: 3, repeat: Infinity, delay: 0.6, ease: "easeInOut" }}
                      />
                    </>
                  )}
                </AnimatePresence>

                <motion.button
                  onClick={toggleRecording}
                  disabled={isLoading}
                  variants={pulseVariants}
                  animate={recording.isRecording ? "recording" : "idle"}
                  className="relative w-[140px] h-[140px] rounded-full flex items-center justify-center"
                  style={{
                    background: recording.isRecording
                      ? 'linear-gradient(135deg, rgba(0, 122, 255, 0.9), rgba(0, 122, 255, 0.7))'
                      : isLoading
                      ? 'linear-gradient(135deg, rgba(255, 184, 0, 0.7), rgba(255, 184, 0, 0.5))'
                      : 'linear-gradient(135deg, rgba(26, 26, 26, 0.9), rgba(26, 26, 26, 0.7))',
                    backdropFilter: 'blur(24px)',
                    WebkitBackdropFilter: 'blur(24px)',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    boxShadow: recording.isRecording 
                      ? '0 16px 64px rgba(0, 122, 255, 0.4)' 
                      : '0 16px 64px rgba(0, 0, 0, 0.24)'
                  }}
                  aria-label={recording.isRecording ? 'Stop recording' : 'Start recording'}
                >
                  <AnimatePresence mode="wait">
                    {recording.isRecording ? (
                      <motion.div
                        key="stop"
                        initial={{ scale: 0, rotate: -180 }}
                        animate={{ scale: 1, rotate: 0 }}
                        exit={{ scale: 0, rotate: 180 }}
                        transition={{ duration: 0.3, ease: bouncyEasing }}
                      >
                        <StopIcon className="h-12 w-12 text-white" />
                      </motion.div>
                    ) : isLoading ? (
                      <motion.div
                        key="loading"
                        initial={{ scale: 0, rotate: -180 }}
                        animate={{ scale: 1, rotate: 0 }}
                        exit={{ scale: 0, rotate: 180 }}
                        transition={{ duration: 0.3, ease: bouncyEasing }}
                      >
                        <ArrowPathIcon className="h-12 w-12 text-white animate-spin" />
                      </motion.div>
                    ) : (
                      <motion.div
                        key="mic"
                        initial={{ scale: 0, rotate: -180 }}
                        animate={{ scale: 1, rotate: 0 }}
                        exit={{ scale: 0, rotate: 180 }}
                        transition={{ duration: 0.3, ease: bouncyEasing }}
                      >
                        <MicrophoneIcon className="h-12 w-12 text-white" />
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.button>
              </motion.div>
            </div>

            {/* Advanced Waveform Visualization */}
            <div className="h-16 mb-8">
              <AnimatePresence mode="wait">
                {recording.isRecording ? (
                  <motion.div 
                    className="flex items-end justify-center h-full gap-[2px]"
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.8 }}
                    transition={{ duration: 0.4, ease: customEasing }}
                  >
                    {[...Array(32)].map((_, i) => (
                      <motion.div
                        key={i}
                        className="w-[3px] rounded-full"
                        custom={i}
                        variants={waveformVariants}
                        animate="listening"
                        style={{
                          transformOrigin: 'bottom',
                          boxShadow: '0 0 8px rgba(0, 122, 255, 0.5)'
                        }}
                      />
                    ))}
                  </motion.div>
                ) : isLoading ? (
                  <motion.div 
                    className="flex items-center justify-center h-full"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                  >
                    <div className="flex items-center gap-4">
                      <motion.div 
                        className="w-6 h-6 border-2 border-t-transparent border-[#FFB800] rounded-full"
                        animate={{ rotate: 360 }}
                        transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                      />
                      <span className="text-[16px] leading-[24px] text-[#FFB800] font-medium">
                        Processing audio...
                      </span>
                    </div>
                  </motion.div>
                ) : speakingAnimation ? (
                  <motion.div 
                    className="flex items-end justify-center h-full gap-[2px]"
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.8 }}
                  >
                    {[...Array(32)].map((_, i) => (
                      <motion.div
                        key={i}
                        className="w-[3px] rounded-full"
                        custom={i}
                        variants={speakingVariants}
                        animate="speaking"
                        style={{
                          transformOrigin: 'bottom',
                          boxShadow: '0 0 8px rgba(0, 208, 132, 0.5)'
                        }}
                      />
                    ))}
                  </motion.div>
                ) : (
                  <motion.div 
                    className="flex items-end justify-center h-full gap-[2px]"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                  >
                    {[...Array(32)].map((_, i) => (
                      <motion.div
                        key={i}
                        className="w-[3px] h-[4px] rounded-full bg-gray-700"
                      />
                    ))}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* Advanced Audio Controls */}
            <div 
              className="rounded-2xl p-5 mb-4"
              style={{
                background: 'rgba(26, 26, 26, 0.6)',
                backdropFilter: 'blur(16px)',
                WebkitBackdropFilter: 'blur(16px)',
                border: '1px solid rgba(255, 255, 255, 0.05)',
                boxShadow: '0 8px 32px rgba(0, 0, 0, 0.18)'
              }}
            >
              <div className="flex items-center justify-between mb-4">
                <span className="text-[16px] leading-[24px] font-medium text-gray-300">Audio Settings</span>
                <motion.button
                  onClick={() => setIsMuted(!isMuted)}
                  variants={buttonVariants}
                  whileHover="hover"
                  whileTap="tap"
                  className="p-2.5 rounded-xl hover:bg-gray-700/30 transition-all duration-200"
                  style={{
                    backdropFilter: 'blur(16px)',
                    WebkitBackdropFilter: 'blur(16px)'
                  }}
                  aria-label={isMuted ? 'Unmute audio' : 'Mute audio'}
                >
                  <motion.div
                    animate={{ scale: isMuted ? [1, 1.1, 1] : 1 }}
                    transition={{ duration: 0.3 }}
                  >
                    {isMuted ? (
                      <SpeakerXMarkIcon className="h-5 w-5 text-[#FF6B6B]" />
                    ) : (
                      <SpeakerWaveIcon className="h-5 w-5 text-gray-300" />
                    )}
                  </motion.div>
                </motion.button>
              </div>
              
              <div className="flex items-center gap-4">
                <span className="text-[12px] leading-[16px] text-gray-400 font-medium">Volume</span>
                <div className="flex-1 relative">
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={volume}
                    onChange={(e) => setVolume(Number(e.target.value))}
                    disabled={isMuted}
                    className="w-full h-2 bg-gray-800 rounded-full appearance-none cursor-pointer transition-all duration-200"
                    style={{
                      background: `linear-gradient(to right, #007AFF ${volume}%, #374151 ${volume}%)`,
                    }}
                    aria-label="Volume control"
                  />
                </div>
                <span className="text-[12px] leading-[16px] text-gray-400 w-10 font-mono">{volume}%</span>
              </div>
            </div>

            {/* Voice Tips */}
            <motion.div
              className="text-center"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.7 }}
            >
              <p className="text-[16px] leading-[24px] text-gray-300 mb-2 font-medium">
                💡 {recording.isRecording ? "Speak clearly and naturally..." : 
                    speakingAnimation ? "Assistant is responding..." :
                    "Tap the microphone to start"}
              </p>
              <p className="text-[12px] leading-[16px] text-gray-500">
                Try: "What are the emergency procedures?" or "How do I file a report?"
              </p>
            </motion.div>
          </div>
        </div>
      </motion.div>

      {/* Accessibility features */}
      <div className="sr-only" aria-live="assertive" aria-atomic="true">
        {recording.isRecording ? "Recording in progress, speak now" : 
          isLoading ? "Processing your voice message, please wait" : 
          speakingAnimation ? "Assistant is speaking" :
          "Ready to record your message"}
      </div>

      {/* Custom scrollbar styles */}
      <style>{`
        .overflow-y-auto::-webkit-scrollbar {
          width: 4px;
        }
        .overflow-y-auto::-webkit-scrollbar-track {
          background: transparent;
        }
        .overflow-y-auto::-webkit-scrollbar-thumb {
          background: rgba(75, 85, 99, 0.3);
          border-radius: 2px;
        }
        .overflow-y-auto::-webkit-scrollbar-thumb:hover {
          background: rgba(75, 85, 99, 0.5);
        }
        
        /* iOS momentum scrolling */
        @supports (-webkit-overflow-scrolling: touch) {
          .overflow-y-auto {
            -webkit-overflow-scrolling: touch;
          }
        }
        
        /* Reduced motion preferences */
        @media (prefers-reduced-motion: reduce) {
          *, ::before, ::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
            scroll-behavior: auto !important;
          }
        }
      `}</style>

      {/* Pending audio play */}
      {pendingAudio && (
        <button
          onClick={() => {
            playAudio(pendingAudio);
            setPendingAudio(null);
          }}
          className="fixed bottom-6 right-6 bg-police-blue text-white p-4 rounded-full shadow-xl"
        >
          ▶️ Play Response
        </button>
      )}
    </div>
  );
};

export default VoiceMode;