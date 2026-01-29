"""
Quick Start Guide
Run this to get started with the Medical Consultation ML System
"""

print("""
╔══════════════════════════════════════════════════════════════╗
║  🏥 Medical Consultation Transcription & Summarization      ║
║     AI-Powered Medical Documentation System                 ║
╚══════════════════════════════════════════════════════════════╝

📚 QUICK START GUIDE
═══════════════════════════════════════════════════════════════

Step 1️⃣: Install Dependencies
─────────────────────────────────────────────────────────────
Run this command in your terminal:

    pip install -r requirements.txt

This will install:
• OpenAI Whisper (speech-to-text)
• HuggingFace Transformers (summarization)
• Streamlit (web interface)
• PyTorch (deep learning)
• And other required packages

Step 2️⃣: Verify Installation
─────────────────────────────────────────────────────────────
Run the test script:

    python test_pipeline.py

This will verify all modules are working correctly.

Step 3️⃣: Run the Application
─────────────────────────────────────────────────────────────
Launch the Streamlit web interface:

    streamlit run app.py

Then open your browser to: http://localhost:8501

Step 4️⃣: Process Audio Files
─────────────────────────────────────────────────────────────
1. Upload a medical consultation audio file (WAV, MP3, etc.)
2. Click "Process Audio File"
3. View the transcript, summary, and prescriptions
4. Download the complete report

═══════════════════════════════════════════════════════════════

📖 DOCUMENTATION
─────────────────────────────────────────────────────────────
Full documentation: README.md
Configuration: config.py
Examples: See examples in README.md

⚙️ SYSTEM REQUIREMENTS
─────────────────────────────────────────────────────────────
• Python 3.8+
• 4GB+ RAM
• 5GB+ disk space
• GPU recommended (but not required)

🔧 CONFIGURATION
─────────────────────────────────────────────────────────────
Edit config.py to customize:
• Speech-to-text model size (tiny, base, small, medium, large)
• Summarization model (BART, T5, Pegasus)
• Output formats (JSON, CSV)
• Processing parameters

🎯 SUPPORTED AUDIO FORMATS
─────────────────────────────────────────────────────────────
• WAV
• MP3
• M4A
• FLAC
• OGG

💡 TIPS FOR BEST RESULTS
─────────────────────────────────────────────────────────────
• Use clear audio with minimal background noise
• Ensure proper microphone quality
• Speak clearly and at a moderate pace
• For longer consultations, consider using larger models

⚠️ IMPORTANT NOTES
─────────────────────────────────────────────────────────────
• First run will download models (~1-2GB)
• Processing time varies by audio length and model size
• GPU acceleration significantly speeds up processing
• This is for research/educational purposes only

🐛 TROUBLESHOOTING
─────────────────────────────────────────────────────────────
If you encounter issues:

1. CUDA/GPU issues → Set device to "cpu" in config.py
2. Memory errors → Use smaller models (tiny/base)
3. Model download fails → Check internet connection
4. Import errors → Reinstall requirements

See README.md for detailed troubleshooting guide.

═══════════════════════════════════════════════════════════════

🚀 Ready to get started?

Run:  streamlit run app.py

═══════════════════════════════════════════════════════════════
""")

# Check if dependencies are installed
print("\n🔍 Checking dependencies...")

try:
    import streamlit
    print("✅ Streamlit installed")
except ImportError:
    print("❌ Streamlit not installed")

try:
    import torch
    print(f"✅ PyTorch installed (CUDA available: {torch.cuda.is_available()})")
except ImportError:
    print("❌ PyTorch not installed")

try:
    import transformers
    print("✅ Transformers installed")
except ImportError:
    print("❌ Transformers not installed")

try:
    import whisper
    print("✅ Whisper installed")
except ImportError:
    try:
        import faster_whisper
        print("✅ Faster-whisper installed")
    except ImportError:
        print("❌ Neither Whisper nor Faster-whisper installed")

print("\n" + "="*60)
print("Installation check complete!")
print("="*60)
print("\nIf any packages are missing, run:")
print("  pip install -r requirements.txt")
print("\nThen run:")
print("  streamlit run app.py")
