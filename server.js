// server.js - Express.js backend for AI model routing with Groq API integration
const express = require('express');
const cors = require('cors');
const axios = require('axios');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const FormData = require('form-data');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Serve uploaded files
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// Configuration for model endpoints using Groq API
const MODEL_CONFIG = {
  balanced: {
    model: process.env.BALANCED_MODEL || 'llama-3.3-70b-versatile',
    endpoint: `${process.env.GROQ_API_BASE_URL || 'https://api.groq.com/openai/v1'}/chat/completions`,
    apiKey: process.env.GROQ_API_KEY,
    provider: 'groq'
  },
  coding: {
    model: process.env.CODING_MODEL || 'qwen-2.5-coder-32b',
    endpoint: `${process.env.GROQ_API_BASE_URL || 'https://api.groq.com/openai/v1'}/chat/completions`,
    apiKey: process.env.GROQ_API_KEY,
    provider: 'groq'
  },
  vision: {
    model: process.env.VISION_MODEL || 'llama-3.2-11b-vision-preview',
    endpoint: `${process.env.GROQ_API_BASE_URL || 'https://api.groq.com/openai/v1'}/chat/completions`,
    apiKey: process.env.GROQ_API_KEY,
    provider: 'groq'
  },
  audio: {
    model: process.env.AUDIO_MODEL || 'whisper-large-v3-turbo',
    endpoint: `${process.env.GROQ_API_BASE_URL || 'https://api.groq.com/openai/v1'}/audio/transcriptions`,
    apiKey: process.env.GROQ_API_KEY,
    provider: 'groq'
  }
};

// Request limiter middleware
const requestLimits = {
  balanced: { perMin: 30, perDay: 14400, tokenLimit: 6000 },
  coding: { perMin: 30, perDay: 1000, tokenLimit: 6000 },
  vision: { perMin: 30, perDay: 7000, tokenLimit: 7000 },
  audio: { perMin: 20, perDay: 2000, secondsLimit: 7200 }
};

// In-memory storage for request tracking (in production, use Redis or similar)
const requestTracker = {
  balanced: { minute: 0, day: 0, lastMinute: Date.now(), lastDay: Date.now() },
  coding: { minute: 0, day: 0, lastMinute: Date.now(), lastDay: Date.now() },
  vision: { minute: 0, day: 0, lastMinute: Date.now(), lastDay: Date.now() },
  audio: { minute: 0, day: 0, lastMinute: Date.now(), lastDay: Date.now() }
};

// Check if request limit is reached
function checkRequestLimit(modelType) {
  const now = Date.now();
  const tracker = requestTracker[modelType];
  const limits = requestLimits[modelType];
  
  // Reset counters if needed
  if (now - tracker.lastMinute > 60000) {
    tracker.minute = 0;
    tracker.lastMinute = now;
  }
  
  if (now - tracker.lastDay > 86400000) {
    tracker.day = 0;
    tracker.lastDay = now;
  }
  
  // Check limits
  if (tracker.minute >= limits.perMin) {
    return { allowed: false, reason: 'Rate limit exceeded: too many requests per minute' };
  }
  
  if (tracker.day >= limits.perDay) {
    return { allowed: false, reason: 'Daily limit exceeded: too many requests per day' };
  }
  
  // Increment counters
  tracker.minute++;
  tracker.day++;
  
  return { allowed: true };
}

// Setup file storage for media uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = path.join(__dirname, 'uploads');
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    cb(null, `${Date.now()}-${file.originalname}`);
  }
});

const upload = multer({ 
  storage,
  limits: { fileSize: 25 * 1024 * 1024 }, // 25MB limit
  fileFilter: (req, file, cb) => {
    // Check file types based on model
    const modelType = req.body.model_type;
    
    if (modelType === 'vision' && !file.mimetype.startsWith('image/')) {
      return cb(new Error('Only image files are allowed for vision models'));
    }
    
    if (modelType === 'audio' && !file.mimetype.startsWith('audio/')) {
      return cb(new Error('Only audio files are allowed for audio models'));
    }
    
    cb(null, true);
  }
});

// Media upload endpoint
app.post('/api/upload', upload.single('media'), (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }
    
    // Return the file path that can be used in the generate endpoint
    const fileUrl = `${req.protocol}://${req.get('host')}/uploads/${req.file.filename}`;
    res.json({ mediaUrl: fileUrl });
  } catch (error) {
    console.error('Error processing upload:', error);
    res.status(500).json({ error: 'Error processing upload', message: error.message });
  }
});

// Main API endpoint for generating AI responses
app.post('/api/generate', async (req, res) => {
  try {
    const { model_type, prompt, max_tokens = 256, media_url } = req.body;
    
    // Validate required fields
    if (!model_type || !prompt) {
      return res.status(400).json({ 
        error: 'Missing required fields', 
        required: ['model_type', 'prompt'] 
      });
    }
    
    // Check if model type is valid
    if (!MODEL_CONFIG[model_type]) {
      return res.status(400).json({ 
        error: 'Invalid model type', 
        valid_types: Object.keys(MODEL_CONFIG) 
      });
    }
    
    // Check request limits
    const limitCheck = checkRequestLimit(model_type);
    if (!limitCheck.allowed) {
      return res.status(429).json({ error: limitCheck.reason });
    }
    
    // Check token limit
    const tokenLimit = requestLimits[model_type].tokenLimit;
    if (tokenLimit && max_tokens > tokenLimit) {
      return res.status(400).json({ 
        error: `Token limit exceeded. Maximum allowed: ${tokenLimit}` 
      });
    }
    
    // Additional validation for media_url
    if ((model_type === 'vision' || model_type === 'audio') && !media_url) {
      return res.status(400).json({ 
        error: `${model_type} model requires a media_url` 
      });
    }
    
    // Prepare request to AI provider
    const modelConfig = MODEL_CONFIG[model_type];
    
    // Different payload structure based on model type and provider
    let payload;
    let response;
    let result;
    
    if (modelConfig.provider === 'groq') {
      switch(model_type) {
        case 'balanced':
        case 'coding':
          payload = {
            model: modelConfig.model,
            messages: [{ role: 'user', content: prompt }],
            max_tokens: max_tokens
          };
          
          response = await axios.post(modelConfig.endpoint, payload, {
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${modelConfig.apiKey}`
            },
            timeout: 60000 // 60 second timeout
          });
          
          result = {
            model: modelConfig.model,
            response: response.data.choices[0].message.content,
            usage: response.data.usage
          };
          break;
          
        case 'vision':
          // Note: Check Groq's documentation for the exact format for vision models
          // This follows a common pattern but might need adjustment
          payload = {
            model: modelConfig.model,
            messages: [
              { 
                role: 'user', 
                content: [
                  { type: 'text', text: prompt },
                  { type: 'image_url', image_url: { url: media_url } }
                ]
              }
            ],
            max_tokens: max_tokens
          };
          
          response = await axios.post(modelConfig.endpoint, payload, {
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${modelConfig.apiKey}`
            },
            timeout: 60000 // 60 second timeout
          });
          
          result = {
            model: modelConfig.model,
            response: response.data.choices[0].message.content,
            usage: response.data.usage
          };
          break;
          
        case 'audio':
          // For audio, we might need to download the file first if it's a URL
          try {
            let audioData;
            let fileName;
            
            if (media_url.startsWith('http')) {
              // Download the file if it's a remote URL
              const audioResponse = await axios.get(media_url, { responseType: 'arraybuffer' });
              audioData = Buffer.from(audioResponse.data);
              fileName = 'audio.mp3'; // Default filename
            } else if (media_url.startsWith('/uploads/')) {
              // If it's a local upload, read from the file system
              const filePath = path.join(__dirname, media_url);
              audioData = fs.readFileSync(filePath);
              fileName = path.basename(filePath);
            } else {
              throw new Error('Invalid media URL format');
            }
            
            // Create form data for the audio API
            const form = new FormData();
            form.append('file', audioData, fileName);
            form.append('model', modelConfig.model);
            if (prompt) form.append('prompt', prompt);
            
            // Send request to the audio API
            response = await axios.post(modelConfig.endpoint, form, {
              headers: {
                ...form.getHeaders(),
                'Authorization': `Bearer ${modelConfig.apiKey}`
              },
              timeout: 120000 // 120 second timeout for audio processing
            });
            
            result = {
              model: modelConfig.model,
              transcription: response.data.text,
              response: response.data.text
            };
          } catch (error) {
            console.error('Error processing audio:', error);
            throw new Error(`Audio processing error: ${error.message}`);
          }
          break;
      }
    } else {
      throw new Error(`Unsupported provider: ${modelConfig.provider}`);
    }
    
    res.json(result);
    
  } catch (error) {
    console.error('Error processing request:', error);
    
    // Format error response
    let errorMessage = error.message || 'Unknown error';
    let statusCode = error.response?.status || 500;
    let errorDetails = error.response?.data || {};
    
    res.status(statusCode).json({
      error: 'Error processing your request',
      message: errorMessage,
      details: errorDetails
    });
  }
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Serve frontend
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Handle 404
app.use((req, res) => {
  res.status(404).json({ error: 'Not found' });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Server error:', err);
  res.status(500).json({ error: 'Server error', message: err.message });
});

// Start server
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`Frontend available at http://localhost:${PORT}`);
  console.log(`API endpoint at http://localhost:${PORT}/api/generate`);
});

module.exports = app; // For testing