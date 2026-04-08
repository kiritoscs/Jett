// DOM elements
const modelSelect = document.getElementById('modelSelect');
const fileDrop = document.getElementById('fileDrop');
const fileInput = document.getElementById('fileInput');
const previewArea = document.getElementById('previewArea');
const previewContent = document.getElementById('previewContent');
const clearBtn = document.getElementById('clearBtn');
const ocrBtn = document.getElementById('ocrBtn');
const loadingSection = document.getElementById('loadingSection');
const resultSection = document.getElementById('resultSection');
const usedModel = document.getElementById('usedModel');
const processingTime = document.getElementById('processingTime');
const ocrResult = document.getElementById('ocrResult');
const copyBtn = document.getElementById('copyBtn');

// State
let selectedFile = null;

// Initialize: load available models
async function loadModels() {
    try {
        const response = await fetch('/api/models');
        const models = await response.json();

        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.id;
            option.textContent = model.name;
            modelSelect.appendChild(option);
        });
    } catch (error) {
        console.error('Failed to load models:', error);
    }
}

// File drop handling
fileDrop.addEventListener('click', () => fileInput.click());

fileDrop.addEventListener('dragover', (e) => {
    e.preventDefault();
    fileDrop.classList.add('dragover');
});

fileDrop.addEventListener('dragleave', () => {
    fileDrop.classList.remove('dragover');
});

fileDrop.addEventListener('drop', (e) => {
    e.preventDefault();
    fileDrop.classList.remove('dragover');
    handleFiles(e.dataTransfer.files);
});

fileInput.addEventListener('change', () => {
    handleFiles(fileInput.files);
});

function handleFiles(files) {
    if (files.length === 0) return;

    selectedFile = files[0];
    showPreview();
    updateUI();
}

function showPreview() {
    previewArea.style.display = 'block';
    clearBtn.style.display = 'inline-block';

    if (selectedFile.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (e) => {
            previewContent.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
        };
        reader.readAsDataURL(selectedFile);
    } else if (selectedFile.type === 'application/pdf') {
        previewContent.innerHTML = `
            <div class="pdf-preview">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                    <polyline points="14 2 14 8 20 8"></polyline>
                    <line x1="16" y1="13" x2="8" y2="13"></line>
                    <line x1="16" y1="17" x2="8" y2="17"></line>
                    <polyline points="10 9 9 9 8 9"></polyline>
                </svg>
                <p>PDF: ${selectedFile.name}</p>
            </div>
        `;
    }
}

function updateUI() {
    const modelSelected = modelSelect.value !== '';
    const fileSelected = selectedFile !== null;
    ocrBtn.disabled = !(modelSelected && fileSelected);
}

modelSelect.addEventListener('change', updateUI);

clearBtn.addEventListener('click', () => {
    selectedFile = null;
    fileInput.value = '';
    previewArea.style.display = 'none';
    clearBtn.style.display = 'none';
    resultSection.style.display = 'none';
    updateUI();
});

ocrBtn.addEventListener('click', async () => {
    if (!selectedFile || !modelSelect.value) return;

    const startTime = Date.now();
    loadingSection.style.display = 'block';
    resultSection.style.display = 'none';
    ocrBtn.disabled = true;

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch(`/api/ocr/${modelSelect.value}`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        const time = ((Date.now() - startTime) / 1000).toFixed(2);

        // Display result
        usedModel.textContent = getModelName(modelSelect.value);
        processingTime.textContent = `${time}s`;

        if (result.text) {
            ocrResult.textContent = result.text;
        } else if (result.pages && Array.isArray(result.pages)) {
            const allText = result.pages.map((p, i) => {
                return `--- 第 ${i + 1} 页 ---\n${p.text}`;
            }).join('\n\n');
            ocrResult.textContent = allText;
        } else {
            ocrResult.textContent = JSON.stringify(result, null, 2);
        }

        resultSection.style.display = 'block';
    } catch (error) {
        ocrResult.textContent = `识别出错: ${error.message}`;
        resultSection.style.display = 'block';
    } finally {
        loadingSection.style.display = 'none';
        ocrBtn.disabled = false;
    }
});

copyBtn.addEventListener('click', async () => {
    const text = ocrResult.textContent;
    try {
        await navigator.clipboard.writeText(text);
        const originalText = copyBtn.textContent;
        copyBtn.textContent = '已复制!';
        setTimeout(() => {
            copyBtn.textContent = originalText;
        }, 2000);
    } catch (error) {
        alert('复制失败，请手动复制');
    }
});

function getModelName(modelId) {
    const option = modelSelect.querySelector(`option[value="${modelId}"]`);
    return option ? option.textContent : modelId;
}

// Load models on page load
document.addEventListener('DOMContentLoaded', loadModels);
