// Main JavaScript for PDF Malware Analyzer

document.addEventListener('DOMContentLoaded', function() {
    initializeFileUpload();
    initializeFormValidation();
    initializeProgressIndicators();
    initializeTooltips();
});

function initializeFileUpload() {
    const fileInput = document.getElementById('file');
    const dropArea = document.getElementById('dropArea');
    const analyzeForm = document.getElementById('analyzeForm');
    
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                if (validateFile(file)) {
                    updateDropAreaWithFile(file);
                    showAnalysisSection(file);
                }
            }
        });
    }
    
    // Initialize drag and drop
    if (dropArea) {
        initializeDragAndDrop(dropArea, fileInput);
    }
    
    // Handle analyze form submission
    if (analyzeForm) {
        analyzeForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const fileInput = document.getElementById('file');
            const file = fileInput.files[0];
            
            if (file) {
                submitAnalysis(file);
            }
        });
    }
}

function initializeDragAndDrop(dropArea, fileInput) {
    // Click to open file dialog
    dropArea.addEventListener('click', function() {
        fileInput.click();
    });
    
    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });
    
    // Highlight drop area when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });
    
    // Handle dropped files
    dropArea.addEventListener('drop', handleDrop, false);
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function highlight(e) {
    const dropArea = document.getElementById('dropArea');
    dropArea.classList.add('dragover');
}

function unhighlight(e) {
    const dropArea = document.getElementById('dropArea');
    dropArea.classList.remove('dragover');
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    const fileInput = document.getElementById('file');
    
    if (files.length > 0) {
        const file = files[0];
        
        // Set the file to the input
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        fileInput.files = dataTransfer.files;
        
        // Validate and update UI
        if (validateFile(file)) {
            updateDropAreaWithFile(file);
        } else {
            showDropAreaError();
        }
    }
}

function updateDropAreaWithFile(file) {
    const dropArea = document.getElementById('dropArea');
    dropArea.classList.add('has-file');
    dropArea.classList.remove('error');
    
    const fileSize = formatFileSize(file.size);
    dropArea.innerHTML = `
        <div class="drop-content">
            <i data-feather="check-circle" class="drop-icon text-success"></i>
            <p class="drop-text text-success">파일이 선택되었습니다</p>
            <div class="file-preview">
                <i data-feather="file-text" class="file-preview-icon"></i>
                <div class="file-preview-info">
                    <div class="file-preview-name">${file.name}</div>
                    <div class="file-preview-size">${fileSize}</div>
                </div>
            </div>
        </div>
    `;
    
    // Re-initialize feather icons
    feather.replace();
}

function showAnalysisSection(file) {
    const analysisSection = document.getElementById('analysisSection');
    analysisSection.style.display = 'block';
    
    // Scroll to analysis section
    analysisSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function submitAnalysis(file) {
    const analyzeBtn = document.getElementById('analyzeBtn');
    const originalText = analyzeBtn.innerHTML;
    
    // Show loading state
    analyzeBtn.innerHTML = `
        <span class="spinner-border spinner-border-sm me-2" role="status"></span>
        분석 중...
    `;
    analyzeBtn.disabled = true;
    
    // Show progress modal
    showProgressModal();
    
    // Create form data
    const formData = new FormData();
    formData.append('file', file);
    
    // Submit analysis
    fetch('/analyze', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (response.ok) {
            return response.text();
        } else {
            return response.json().then(data => {
                throw new Error(data.error || '분석 중 오류가 발생했습니다');
            });
        }
    })
    .then(html => {
        // Replace entire page content with result
        document.open();
        document.write(html);
        document.close();
    })
    .catch(error => {
        // Hide progress modal
        const modal = document.getElementById('progressModal');
        if (modal) {
            const bsModal = bootstrap.Modal.getInstance(modal);
            if (bsModal) {
                bsModal.hide();
            }
        }
        
        // Reset button
        analyzeBtn.innerHTML = originalText;
        analyzeBtn.disabled = false;
        
        // Show error
        showAlert(error.message, 'danger');
    });
}

function showDropAreaError() {
    const dropArea = document.getElementById('dropArea');
    dropArea.classList.add('error');
    dropArea.classList.remove('has-file');
    
    // Reset to original state after animation
    setTimeout(() => {
        dropArea.classList.remove('error');
    }, 1000);
}

function validateFile(file) {
    const maxSize = 16 * 1024 * 1024; // 16MB
    const allowedTypes = ['application/pdf'];
    const allowedExtensions = ['.pdf'];
    
    // Check file type and extension
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
    if (!allowedTypes.includes(file.type) && !allowedExtensions.includes(fileExtension)) {
        showAlert('PDF 파일만 업로드할 수 있습니다', 'danger');
        resetFileInput();
        return false;
    }
    
    // Check file size
    if (file.size > maxSize) {
        showAlert('파일이 너무 큽니다. 최대 크기는 16MB입니다.', 'danger');
        resetFileInput();
        return false;
    }
    
    return true;
}

function resetFileInput() {
    const fileInput = document.getElementById('file');
    const dropArea = document.getElementById('dropArea');
    
    fileInput.value = '';
    
    if (dropArea) {
        dropArea.classList.remove('has-file', 'error');
        dropArea.innerHTML = `
            <div class="drop-content">
                <i data-feather="upload-cloud" class="drop-icon"></i>
                <p class="drop-text">Drop your PDF file here or <span class="text-primary">click to browse</span></p>
                <p class="drop-subtext">Only PDF files • Max 16MB</p>
            </div>
        `;
        feather.replace();
    }
}

function showFileInfo(file) {
    const fileSize = formatFileSize(file.size);
    const fileName = file.name;
    
    // Create or update file info display
    let fileInfoDiv = document.getElementById('fileInfo');
    if (!fileInfoDiv) {
        fileInfoDiv = document.createElement('div');
        fileInfoDiv.id = 'fileInfo';
        fileInfoDiv.className = 'mt-3 p-3 bg-light rounded';
        document.getElementById('file').parentNode.appendChild(fileInfoDiv);
    }
    
    fileInfoDiv.innerHTML = `
        <div class="d-flex align-items-center">
            <i data-feather="file-text" class="me-2 text-primary"></i>
            <div>
                <strong>${fileName}</strong><br>
                <small class="text-muted">${fileSize}</small>
            </div>
        </div>
    `;
    
    // Re-initialize feather icons
    feather.replace();
}

function showUploadProgress(button) {
    const originalText = button.innerHTML;
    button.innerHTML = `
        <span class="spinner-border spinner-border-sm me-2" role="status"></span>
        Uploading...
    `;
    button.disabled = true;
    
    // Reset button state after 30 seconds (failsafe)
    setTimeout(() => {
        button.innerHTML = originalText;
        button.disabled = false;
    }, 30000);
}

function initializeFormValidation() {
    const forms = document.querySelectorAll('.needs-validation');
    
    Array.from(forms).forEach(form => {
        form.addEventListener('submit', event => {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        });
    });
}

function initializeProgressIndicators() {
    const analyzeBtn = document.getElementById('analyzeBtn');
    
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', function(e) {
            showAnalysisProgress(this);
        });
    }
}

function showAnalysisProgress(button) {
    const originalText = button.innerHTML;
    button.innerHTML = `
        <span class="spinner-border spinner-border-sm me-2" role="status"></span>
        Analyzing...
    `;
    button.disabled = true;
    
    // Show progress modal or indicator
    showProgressModal();
}

function showProgressModal() {
    // Create a simple progress modal
    const modalHtml = `
        <div class="modal fade" id="progressModal" tabindex="-1" data-bs-backdrop="static">
            <div class="modal-dialog modal-dialog-centered">
                <div class="modal-content">
                    <div class="modal-body text-center p-4">
                        <div class="spinner-border text-primary mb-3" role="status">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                        <h5>Analyzing PDF...</h5>
                        <p class="text-muted mb-0">Please wait while we scan your file for malware</p>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Add modal to page if it doesn't exist
    if (!document.getElementById('progressModal')) {
        document.body.insertAdjacentHTML('beforeend', modalHtml);
    }
    
    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('progressModal'));
    modal.show();
}

function initializeTooltips() {
    // Initialize Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

function showAlert(message, type = 'info') {
    const alertHtml = `
        <div class="alert alert-${type} alert-dismissible fade show" role="alert">
            <i data-feather="${type === 'danger' ? 'alert-circle' : 'info'}" class="me-2"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    
    // Insert alert at the top of the main container
    const container = document.querySelector('main.container');
    if (container) {
        container.insertAdjacentHTML('afterbegin', alertHtml);
        feather.replace();
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            const alert = container.querySelector('.alert');
            if (alert) {
                const bsAlert = new bootstrap.Alert(alert);
                bsAlert.close();
            }
        }, 5000);
    }
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Handle page reload confirmation for analysis in progress
window.addEventListener('beforeunload', function(e) {
    const analyzeBtn = document.getElementById('analyzeBtn');
    if (analyzeBtn && analyzeBtn.disabled) {
        e.preventDefault();
        e.returnValue = 'Analysis is in progress. Are you sure you want to leave?';
        return e.returnValue;
    }
});

// Auto-refresh feather icons after dynamic content updates
function refreshFeatherIcons() {
    if (typeof feather !== 'undefined') {
        feather.replace();
    }
}

// Utility function to copy text to clipboard
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(function() {
        showAlert('Copied to clipboard!', 'success');
    }).catch(function(err) {
        console.error('Could not copy text: ', err);
        showAlert('Failed to copy to clipboard', 'danger');
    });
}

// Add keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl+U or Cmd+U for upload (when no modal is open)
    if ((e.ctrlKey || e.metaKey) && e.key === 'u') {
        e.preventDefault();
        const fileInput = document.getElementById('file');
        if (fileInput && !document.querySelector('.modal.show')) {
            fileInput.click();
        }
    }
    
    // Escape to close modals
    if (e.key === 'Escape') {
        const modal = document.querySelector('.modal.show');
        if (modal) {
            const bsModal = bootstrap.Modal.getInstance(modal);
            if (bsModal) {
                bsModal.hide();
            }
        }
    }
});
