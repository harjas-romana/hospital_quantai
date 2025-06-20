## 🚀 Features
### Core Functionality
- **Text Mode**: Type questions and receive responses in a chat-like interface
- **Voice Mode**: Interact via voice input/output with visual feedback during recording
- **Real-time Chat**: Smooth message animations and auto-scroll functionality
- **Loading States**: Animated loading indicators and typing indicators

### Advanced UI/UX
- **Smooth Animations**: Powered by Framer Motion for fluid transitions
- **Interactive Charts**: Data visualization with Recharts library
- **Toast Notifications**: User feedback with React Hot Toast
- **Settings Modal**: Comprehensive settings with accessibility options
- **Responsive Design**: Works seamlessly on desktop and mobile devices

### Accessibility & Performance
- **ARIA Labels**: Full accessibility support for screen readers
- **Keyboard Navigation**: Complete keyboard accessibility
- **Focus Management**: Proper focus handling for all interactive elements
- **Performance Optimized**: Code-splitting and lazy loading for heavy components

### Data Visualization
- **Response Time Charts**: Area charts showing AI response performance
- **Query Distribution**: Pie charts displaying query type breakdowns
- **Usage Analytics**: Bar charts for daily query volumes
- **Key Metrics**: Real-time performance indicators

## 🛠 Tech Stack

- **React.js** with TypeScript
- **Vite** for fast development and building
- **Tailwind CSS v4** for styling
- **Framer Motion** for animations
- **React Hot Toast** for notifications
- **Recharts** for data visualization
- **Headless UI** for accessible components
- **Heroicons** for animated SVG icons

## 📦 Installation

### Prerequisites
- Node.js 16+ and npm/yarn
- Modern web browser with ES6+ support

### Setup Instructions

1. **Clone and Navigate**
   ```bash
   cd frontend
   ```

2. **Install Dependencies**
   ```bash
   npm install
   ```

3. **Start Development Server**
   ```bash
   npm run dev
   ```

4. **Build for Production**
   ```bash
   npm run build
   ```

## 🎨 Key Components

### Animation System
- **Framer Motion Integration**: Smooth page transitions and component animations
- **Custom Animation Variants**: Reusable animation patterns
- **Performance Optimized**: GPU-accelerated animations for smooth 60fps

### Data Visualization
- **Interactive Charts**: Hover effects and tooltips
- **Responsive Design**: Charts adapt to container size
- **Theme Integration**: Consistent with application color scheme

### Accessibility Features
- **Screen Reader Support**: Comprehensive ARIA labels
- **Keyboard Navigation**: Full keyboard accessibility
- **Focus Indicators**: Clear visual focus states
- **Reduced Motion**: Respects user motion preferences

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the frontend directory:
```env
VITE_APP_TITLE=Australian Police Department AI Assistant
VITE_APP_API_URL=http://localhost:8000/api
```

### Tailwind Configuration
Custom animations and colors are defined in `tailwind.config.js`:
- Custom police blue and gold colors
- Animation keyframes for fade-in, slide-in, and pulse effects
- Responsive breakpoints

## 📱 Usage

### Text Mode
1. Type your question in the input field
2. Press Enter or click Send
3. View animated response with typing indicator
4. Messages auto-scroll to bottom

### Voice Mode
1. Click the microphone button to start recording
2. Speak your question clearly
3. Click again to stop recording
4. View processing animation and response

### Analytics View
1. Click the Analytics button in the header
2. View interactive charts and metrics
3. Hover over chart elements for detailed information

### Settings
1. Click the Settings button in the header
2. Configure theme, notifications, and accessibility options
3. Toggle switches with smooth animations

## 🎯 Performance Optimizations

- **Code Splitting**: Lazy loading for heavy components
- **Animation Performance**: GPU-accelerated transforms
- **Bundle Optimization**: Tree shaking and minification
- **Image Optimization**: Optimized logo and assets

## 🔒 Security Considerations

- **Input Validation**: Client-side validation for all inputs
- **XSS Prevention**: Sanitized user inputs
- **HTTPS Ready**: Secure communication protocols
- **Content Security Policy**: CSP headers for production

## 📊 Analytics & Monitoring

The application includes built-in analytics features:
- Response time tracking
- Query type distribution
- Usage patterns
- Performance metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with proper TypeScript types
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is a Proof of Concept by QuantAI, NZ. Not intended for production use.

## 🆘 Support

For technical support or questions:
- Check the console for error messages
- Verify all dependencies are installed
- Ensure Node.js version compatibility
- Review browser console for any issues

---

**Built with ❤️ by QuantAI, NZ** 