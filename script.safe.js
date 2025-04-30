// تحديث دالة analyzeImage
async function analyzeImage(imageData) {
    try {
        promptResult.innerHTML = '<div class="loading">جاري تحليل الصورة...</div>';
        promptResult.style.display = 'block';

        const base64Data = imageData.split(',')[1];
        const byteCharacters = atob(base64Data);
        const byteNumbers = new Array(byteCharacters.length);
        
        for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        
        const byteArray = new Uint8Array(byteNumbers);
        const blob = new Blob([byteArray], { type: 'image/jpeg' });

        const formData = new FormData();
        formData.append('image', blob);

        const response = await fetch('http://localhost:3000/analyze-image', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('فشل في الاتصال بالخادم');
        }

        const result = await response.json();
        
        if (result && result[0] && result[0].generated_text) {
            // باقي الكود كما هو...
        }
    } catch (error) {
        promptResult.innerHTML = \`
            <div class="result-box" style="color: #721c24; background-color: #f8d7da; border-color: #f5c6cb;">
                حدث خطأ أثناء تحليل الصورة: \${error.message}
            </div>
        \`;
        console.error('Error:', error);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const imagePreview = document.getElementById('imagePreview');
    const promptResult = document.getElementById('promptResult');
    const optionsPanel = document.getElementById('optionsPanel');

    // 🔒 لم يعد يتم تضمين المفتاح مباشرة هنا

    // إعداد خيارات النمط
    const styleOptions = document.querySelectorAll('.style-option');
    styleOptions.forEach(option => {
        option.addEventListener('click', () => {
            styleOptions.forEach(opt => opt.classList.remove('active'));
            option.classList.add('active');
            if (imagePreview.src) {
                analyzeImage(imagePreview.src);
            }
        });
    });

    const detailOptions = document.querySelectorAll('.detail-options input');
    detailOptions.forEach(option => {
        option.addEventListener('change', () => {
            if (imagePreview.src) {
                analyzeImage(imagePreview.src);
            }
        });
    });

    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#007bff';
    });
    dropZone.addEventListener('dragleave', () => {
        dropZone.style.borderColor = '#ccc';
    });
    dropZone.addEventListener('drop', handleDrop);
    fileInput.addEventListener('change', handleFileSelect);

    function handleDrop(e) {
        e.preventDefault();
        dropZone.style.borderColor = '#ccc';
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    function handleFileSelect(e) {
        const files = e.target.files;
        handleFiles(files);
    }

    function handleFiles(files) {
        if (files.length === 0) return;
        
        const file = files[0];
        if (!file.type.startsWith('image/')) {
            alert('الرجاء اختيار ملف صورة فقط');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.style.display = 'block';
            optionsPanel.style.display = 'block';
            analyzeImage(e.target.result);
        };
        reader.readAsDataURL(file);
    }
});