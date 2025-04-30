require('dotenv').config();
const express = require('express');
const cors = require('cors');
const multer = require('multer');
const fetch = require('node-fetch');

const app = express();
const upload = multer();

app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// نقطة النهاية لتحليل الصورة
app.post('/analyze-image', upload.single('image'), async (req, res) => {
    try {
        const response = await fetch(
            "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large",
            {
                headers: {
                    'Authorization': `Bearer ${process.env.HF_API_KEY}`
                },
                method: "POST",
                body: req.file.buffer
            }
        );

        if (!response.ok) {
            throw new Error('فشل في الاتصال بخدمة تحليل الصور');
        }

        const result = await response.json();
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});