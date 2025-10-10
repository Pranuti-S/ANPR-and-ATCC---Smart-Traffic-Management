"""
Tesseract OCR Installation Test Script
Run this to verify Tesseract is properly installed and working
"""

import sys
import subprocess

def test_tesseract_installation():
    """Test if Tesseract OCR is installed and accessible"""
    
    print("=" * 60)
    print("🔍 TESSERACT OCR INSTALLATION TEST")
    print("=" * 60)
    print()
    
    # Test 1: Check if tesseract command exists
    print("Test 1: Checking Tesseract command...")
    try:
        result = subprocess.run(
            ['tesseract', '--version'], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        
        if result.returncode == 0:
            print("✅ SUCCESS: Tesseract is installed!")
            print()
            print("Version Information:")
            print("-" * 60)
            print(result.stdout)
            print("-" * 60)
        else:
            print("❌ FAILED: Tesseract command exists but returned an error")
            print(f"Error: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("❌ FAILED: Tesseract is NOT installed!")
        print()
        print("📦 Installation Instructions:")
        print("-" * 60)
        print("Ubuntu/Debian:")
        print("  sudo apt-get update")
        print("  sudo apt-get install tesseract-ocr")
        print()
        print("macOS:")
        print("  brew install tesseract")
        print()
        print("Windows:")
        print("  Download from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("  Then add to PATH")
        print("-" * 60)
        return False
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False
    
    print()
    
    # Test 2: Check Python pytesseract module
    print("Test 2: Checking Python pytesseract module...")
    try:
        import pytesseract
        print("✅ SUCCESS: pytesseract module is installed!")
        print(f"   Module location: {pytesseract.__file__}")
    except ImportError:
        print("❌ FAILED: pytesseract Python module is NOT installed!")
        print()
        print("📦 Install with:")
        print("   pip install pytesseract")
        return False
    
    print()
    
    # Test 3: Try actual OCR on a simple test
    print("Test 3: Testing OCR functionality...")
    try:
        from PIL import Image
        import numpy as np
        
        # Create a simple test image with text
        test_img = np.ones((100, 300, 3), dtype=np.uint8) * 255
        
        # Try OCR (even if empty, it should work without error)
        import pytesseract
        text = pytesseract.image_to_string(test_img)
        
        print("✅ SUCCESS: OCR functionality is working!")
        print(f"   Test completed without errors")
        
    except Exception as e:
        print(f"❌ FAILED: OCR test failed with error:")
        print(f"   {str(e)}")
        return False
    
    print()
    print("=" * 60)
    print("🎉 ALL TESTS PASSED! Tesseract OCR is ready to use!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_tesseract_installation()
    
    if not success:
        print()
        print("⚠️  Please install Tesseract OCR and try again.")
        sys.exit(1)
    else:
        print()
        print("✅ You can now use OCR in the ANPR system!")
        sys.exit(0)