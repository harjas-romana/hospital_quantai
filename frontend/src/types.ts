export interface Message {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: Date;
}

export type Mode = 'text' | 'voice';

export interface Recording {
  isRecording: boolean;
  audioURL?: string;
} 