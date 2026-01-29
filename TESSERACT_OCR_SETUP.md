# Tesseract OCR Setup Guide

## What is Tesseract OCR?

Tesseract is an open-source Optical Character Recognition (OCR) engine that extracts text from images. It's required for processing:
- Prescription images (photos, scans)
- Medical document images
- Any image-based document uploads

## Installation Status

✅ **Tesseract OCR v5.4.0** is now installed on your system!

## Automatic Configuration

The application has been configured to automatically find Tesseract in these locations:
1. `C:\Program Files\Tesseract-OCR\tesseract.exe`
2. `C:\Program Files (x86)\Tesseract-OCR\tesseract.exe`
3. `%USERPROFILE%\AppData\Local\Programs\Tesseract-OCR\tesseract.exe`

## Verification

To verify Tesseract is working, open PowerShell and run:
```powershell
tesseract --version
```

You should see:
```
tesseract v5.4.0.20240606
```

## Usage in Application

### Backend - Document Processing

The system now automatically uses Tesseract for:

1. **Prescription Image Upload** (`/api/document/upload/prescription`)
   - PNG, JPG, JPEG, TIFF, BMP images
   - Extracts text using OCR
   - Passes to prescription extractor

2. **Document Analysis** (`/api/document/analyze`)
   - Medical records as images
   - Lab reports
   - EHR documents

### Error Handling

If Tesseract is not found, the application will return:
```json
{
  "detail": "tesseract is not installed or it's not in your PATH"
}
```

**Solution:** Restart the application using `start_app.bat` (which now includes Tesseract in PATH)

## Supported Image Formats

- ✅ PNG
- ✅ JPG/JPEG
- ✅ TIFF
- ✅ BMP
- ✅ PDF (with image content)

## Python Packages

Required packages (already installed):
- `pytesseract` (0.3.13) - Python wrapper for Tesseract
- `Pillow` (11.3.0) - Image processing library

## Troubleshooting

### Error: "tesseract is not installed"

**Cause:** Tesseract not in system PATH

**Solutions:**
1. Run application with `start_app.bat` (recommended)
2. Or run `setup_tesseract.bat` before starting
3. Or add to PATH manually:
   ```powershell
   $env:PATH += ";C:\Program Files\Tesseract-OCR"
   ```

### Error: "Failed to extract text from image"

**Possible causes:**
1. Image quality too low
2. Handwritten text (Tesseract works best with printed text)
3. Image too small or pixelated
4. Wrong language (default is English)

**Solutions:**
- Use higher quality scans/photos
- Ensure good lighting for photos
- Use flat, non-skewed images
- For handwritten prescriptions, consider manual entry

### Reinstallation

If needed, reinstall Tesseract:
```powershell
winget uninstall --id UB-Mannheim.TesseractOCR
winget install --id UB-Mannheim.TesseractOCR
```

## Testing OCR

Test Tesseract from command line:
```powershell
tesseract test_image.png output.txt
```

This will create `output.txt` with extracted text from `test_image.png`.

## Performance Notes

- OCR processing takes 2-5 seconds per image
- Higher resolution = better accuracy but slower processing
- Recommended: 300 DPI for scanned documents
- For best results: Clear, high-contrast images with printed text

## Advanced Configuration

### Language Support

Tesseract supports multiple languages. To use other languages:

1. Download language data from: https://github.com/tesseract-ocr/tessdata
2. Place `.traineddata` files in: `C:\Program Files\Tesseract-OCR\tessdata\`
3. Use in code: `pytesseract.image_to_string(image, lang='spa')` (for Spanish)

### Custom Configuration

Modify in `backend/utils/document_processor.py`:
```python
# Add custom config
config = '--psm 6 --oem 3'  # Page segmentation mode 6, OCR engine mode 3
text = pytesseract.image_to_string(image, lang='eng', config=config)
```

## Integration Notes

The following files have been updated to support Tesseract:
- ✅ `backend/utils/document_processor.py` - Auto-detect Tesseract path
- ✅ `backend/routes/document_analysis.py` - Auto-detect Tesseract path
- ✅ `start_app.bat` - Adds Tesseract to PATH on startup

## Summary

✅ **Tesseract OCR is now installed and configured!**

The error you encountered should now be resolved. The application will automatically find and use Tesseract for image-based document processing.

**To use:** Simply restart your application with `start_app.bat` and try uploading a prescription image again.
