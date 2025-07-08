// AI Medical Coding - Spreadsheet Interface JavaScript - Modular & Clean

// NEW: File Tree Processor for folder uploads
class FileTreeProcessor {
    constructor() {
        this.supportedTypes = ['.txt', '.pdf', '.doc', '.docx', '.html', '.htm'];
    }

    scanFiles(fileList) {
        const processedFiles = [];
        
        for (const file of fileList) {
            const extension = '.' + file.name.split('.').pop().toLowerCase();
            
            if (this.supportedTypes.includes(extension)) {
                const fileInfo = {
                    file: file,
                    name: file.name,
                    size: file.size,
                    relativePath: file.webkitRelativePath || file.name,
                    folderDepth: this.calculateFolderDepth(file.webkitRelativePath || file.name),
                    parentFolder: this.extractParentFolder(file.webkitRelativePath || file.name)
                };
                processedFiles.push(fileInfo);
            }
        }
        
        return processedFiles;
    }

    calculateFolderDepth(path) {
        return path.split('/').length - 1;
    }

    extractParentFolder(path) {
        const parts = path.split('/');
        return parts.length > 1 ? parts[parts.length - 2] : 'root';
    }

    extractRelativePath(file) {
        if (file.webkitRelativePath) {
            // For folder uploads, remove the root folder from the path
            const pathParts = file.webkitRelativePath.split('/');
            return pathParts.slice(1).join('/') || file.name;
        }
        return file.name;
    }
}

class DataProcessor {
    static cleanAndMapResult(rawResult, filepath) {
        // Force correct field mapping with explicit fallbacks
        // IMPORTANT: filepath comes from frontend (full path), title comes from backend (processed from filename only)
        return {
            filepath: filepath, // Full path from frontend
            title: rawResult.title || 'N/A', // Processed title from backend
            gender: rawResult.gender || 'N/A',
            unique_name: rawResult.unique_name || 'N/A',
            keywords: rawResult.keywords || 'N/A',
            icd_code_root: rawResult.icd_code_root || '',
            icd_code_hierarchy: rawResult.icd_code_hierarchy || '',
            details_description: rawResult.details_description || '',
            details_score: rawResult.details_score || '',
            language: rawResult.language || 'English',
            status: 'completed',
            // Keep for backward compatibility
            diagnosis_codes: rawResult.diagnosis_codes || rawResult.icd_code_hierarchy || '',
            cpt_codes: rawResult.cpt_codes || ''
        };
    }

    static formatGenderBadge(gender) {
        if (!gender || gender === 'N/A') return 'N/A';
        return `<span class="gender-badge gender-${gender.toLowerCase()}">${gender}</span>`;
    }

    static formatStatusIcon(isSuccess) {
        return isSuccess 
            ? '<i class="fas fa-check-circle status-completed"></i>'
            : '<i class="fas fa-exclamation-circle status-error"></i>';
    }
}

class TableRenderer {
    static createTableRow(result, isSuccess = true) {
        const statusIcon = DataProcessor.formatStatusIcon(isSuccess);
        const genderBadge = DataProcessor.formatGenderBadge(result.gender);
        
        return `
            <td class="col-filepath">${result.filepath}</td>
            <td class="col-title">${result.title}</td>
            <td class="col-gender">${genderBadge}</td>
            <td class="col-unique-name">${result.unique_name}</td>
            <td class="col-keywords">${result.keywords}</td>
            <td class="col-icd-root">${result.icd_code_root}</td>
            <td class="col-icd-hierarchy">${result.icd_code_hierarchy}</td>
            <td class="col-details-description">${result.details_description}</td>
            <td class="col-details-score">${result.details_score}</td>
            <td class="col-language">${result.language}</td>
            <td class="col-status">${statusIcon} ${result.status}</td>
        `;
    }

    static createPlaceholderRow(filename) {
        return `
            <td class="col-filepath">${filename}</td>
            <td colspan="10" style="color: var(--warning-color); padding: 20px; text-align: center;">
                <div class="spinner" style="display: inline-block; vertical-align: middle; margin-right: 8px;"></div>
                <span style="font-size: 0.9rem; font-weight: 500;">Waiting to process...</span>
            </td>
        `;
    }

    static createProcessingRow(filename) {
        return `
            <td class="col-filepath">${filename}</td>
            <td colspan="9" style="color: var(--warning-color); padding: 20px; text-align: center;">
                <div class="spinner" style="display: inline-block; vertical-align: middle; margin-right: 8px;"></div>
                <span style="font-size: 0.9rem; font-weight: 500;">Processing document...</span>
            </td>
            <td class="col-status"><i class="fas fa-cogs status-processing"></i> Processing</td>
        `;
    }
}

class ExportManager {
    static convertToCSV(data) {
        if (data.length === 0) return '';

        const headers = [
            'FilePath', 'Title', 'Gender', 'Unique Name', 'Keywords', 
            'ICD Code Root', 'ICD Code Hierarchy', 'Details - Description', 
            'Details - Score', 'Language', 'Status'
        ];
        const csvRows = [headers.join(',')];

        data.forEach(row => {
            const values = [
                ExportManager.escapeCSVField(row.filepath),
                ExportManager.escapeCSVField(row.title),
                ExportManager.escapeCSVField(row.gender),
                ExportManager.escapeCSVField(row.unique_name),
                ExportManager.escapeCSVField(row.keywords),
                ExportManager.escapeCSVField(row.icd_code_root),
                ExportManager.escapeCSVField(row.icd_code_hierarchy),
                ExportManager.escapeCSVField(row.details_description),
                ExportManager.escapeCSVField(row.details_score),
                ExportManager.escapeCSVField(row.language),
                ExportManager.escapeCSVField(row.status)
            ];
            csvRows.push(values.join(','));
        });

        return csvRows.join('\n');
    }

    static escapeCSVField(field) {
        if (field === null || field === undefined) return '';
        const stringField = String(field);
        if (stringField.includes(',') || stringField.includes('"') || stringField.includes('\n')) {
            return `"${stringField.replace(/"/g, '""')}"`;
        }
        return stringField;
    }

    static downloadFile(content, filename, mimeType) {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }

    static exportAsExcel(data) {
        const workbook = XLSX.utils.book_new();
        const worksheet = XLSX.utils.json_to_sheet(data);
        
        const colWidths = [
            { wch: 30 }, // filepath
            { wch: 40 }, // title
            { wch: 10 }, // gender
            { wch: 30 }, // unique_name
            { wch: 50 }, // keywords
            { wch: 15 }, // icd_code_root
            { wch: 30 }, // icd_code_hierarchy
            { wch: 60 }, // details_description
            { wch: 30 }, // details_score
            { wch: 10 }, // language
            { wch: 15 }  // status
        ];
        worksheet['!cols'] = colWidths;

        XLSX.utils.book_append_sheet(workbook, worksheet, 'Medical Coding Results');
        XLSX.writeFile(workbook, 'medical_coding_results.xlsx');
    }
}

class NotificationManager {
    static show(message, type = 'info') {
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            border-radius: 10px;
            color: white;
            font-weight: 600;
            z-index: 10000;
            max-width: 300px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            animation: slideInRight 0.3s ease;
        `;

        const colors = {
            success: '#059669',
            error: '#dc2626',
            warning: '#d97706',
            info: '#2563eb'
        };

        notification.style.background = colors[type] || colors.info;
        notification.textContent = message;

        document.body.appendChild(notification);

        setTimeout(() => {
            notification.style.animation = 'slideOutRight 0.3s ease';
            setTimeout(() => {
                if (notification.parentNode) {
                    document.body.removeChild(notification);
                }
            }, 300);
        }, 3000);
    }
}

class SpreadsheetProcessor {
    constructor() {
        this.files = [];
        this.results = [];
        this.isProcessing = false;
        this.currentFileIndex = 0;
        this.completedFiles = 0;
        this.currentMode = 'files'; // 'files' or 'folder'
        this.fileTreeProcessor = new FileTreeProcessor();
        
        this.initializeElements();
        this.bindEvents();
    }

    initializeElements() {
        // Main elements
        this.fileInput = document.getElementById('fileInput');
        this.folderInput = document.getElementById('folderInput');
        this.uploadArea = document.getElementById('uploadArea');
        this.fileList = document.getElementById('fileList');
        
        // Mode toggle elements
        this.modeButtons = document.querySelectorAll('.mode-btn');
        this.uploadIcon = document.getElementById('uploadIcon');
        this.uploadText = document.getElementById('uploadText');
        this.uploadHint = document.getElementById('uploadHint');
        
        // Buttons
        this.selectFilesBtn = document.getElementById('selectFilesBtn');
        this.processBtn = document.getElementById('processBtn');
        this.clearBtn = document.getElementById('clearBtn');
        this.exportBtn = document.getElementById('exportBtn');
        this.newBatchBtn = document.getElementById('newBatchBtn');
        
        // Progress elements
        this.progressContainer = document.getElementById('progressContainer');
        this.progressBar = document.getElementById('progressBar');
        this.progressStats = document.getElementById('progressStats');
        this.currentProcessing = document.getElementById('currentProcessing');
        
        // Results elements
        this.resultsSection = document.getElementById('resultsSection');
        this.resultsTableBody = document.getElementById('resultsTableBody');
        this.totalFiles = document.getElementById('totalFiles');
        this.successCount = document.getElementById('successCount');
        this.errorCount = document.getElementById('errorCount');
        this.totalCodes = document.getElementById('totalCodes');
        
        // Modal elements
        this.exportModal = document.getElementById('exportModal');
        this.closeModal = document.querySelector('.close');
        this.exportCSV = document.getElementById('exportCSV');
        this.exportExcel = document.getElementById('exportExcel');
        this.exportJSON = document.getElementById('exportJSON');
    }

    bindEvents() {
        // Mode toggle events
        this.modeButtons.forEach(btn => {
            btn.addEventListener('click', (e) => this.switchMode(e.target.dataset.mode));
        });
        
        // File selection events
        this.selectFilesBtn.addEventListener('click', () => this.selectFiles());
        this.fileInput.addEventListener('change', (e) => this.handleFileSelection(e));
        this.folderInput.addEventListener('change', (e) => this.handleFolderSelection(e));
        
        // Drag and drop events
        this.uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        this.uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        this.uploadArea.addEventListener('drop', (e) => this.handleDrop(e));
        
        // Control buttons
        this.processBtn.addEventListener('click', () => this.startProcessing());
        this.clearBtn.addEventListener('click', () => this.clearAll());
        this.newBatchBtn.addEventListener('click', () => this.newBatch());
        
        // Export events
        this.exportBtn.addEventListener('click', () => this.showExportModal());
        this.closeModal.addEventListener('click', () => this.hideExportModal());
        this.exportCSV.addEventListener('click', () => this.exportAsCSV());
        this.exportExcel.addEventListener('click', () => this.exportAsExcel());
        this.exportJSON.addEventListener('click', () => this.exportAsJSON());
        
        // Modal close on outside click
        window.addEventListener('click', (e) => {
            if (e.target === this.exportModal) {
                this.hideExportModal();
            }
        });
    }

    switchMode(mode) {
        this.currentMode = mode;
        
        // Update button states
        this.modeButtons.forEach(btn => {
            btn.classList.remove('active');
            if (btn.dataset.mode === mode) {
                btn.classList.add('active');
                btn.style.background = 'var(--primary-color)';
                btn.style.color = 'white';
            } else {
                btn.style.background = 'transparent';
                btn.style.color = 'var(--gray-600)';
            }
        });
        
        // Update UI elements
        if (mode === 'folder') {
            this.fileInput.style.display = 'none';
            this.folderInput.style.display = 'block';
            this.uploadIcon.className = 'fas fa-folder-open';
            this.uploadText.textContent = 'Drag & Drop Folder Here';
            this.uploadHint.innerHTML = `
                or click to browse<br>
                <small>Supports: PDF, DOC, DOCX, TXT, HTML | Entire folder structure</small><br>
                <strong style="color: var(--primary-color);">Will process all supported files recursively</strong>
            `;
            this.selectFilesBtn.innerHTML = '<i class="fas fa-folder"></i> Select Folder';
        } else {
            this.fileInput.style.display = 'block';
            this.folderInput.style.display = 'none';
            this.uploadIcon.className = 'fas fa-file-medical';
            this.uploadText.textContent = 'Drag & Drop Files Here';
            this.uploadHint.innerHTML = `
                or click to browse<br>
                <small>Supports: PDF, DOC, DOCX, TXT, HTML | Select multiple files</small><br>
                <strong style="color: var(--primary-color);">Expected filename format:</strong><br>
                <code style="background: #f0f0f0; padding: 2px 4px; border-radius: 3px;">Title Name 05-24-2025.pdf</code> or 
                <code style="background: #f0f0f0; padding: 2px 4px; border-radius: 3px;">Title Name.html</code>
            `;
            this.selectFilesBtn.innerHTML = '<i class="fas fa-file-medical"></i> Select Files';
        }
    }

    selectFiles() {
        if (this.currentMode === 'folder') {
            this.folderInput.click();
        } else {
            this.fileInput.multiple = true;
            this.fileInput.click();
        }
    }

    handleFileSelection(event) {
        const selectedFiles = Array.from(event.target.files);
        this.addFiles(selectedFiles);
    }

    handleFolderSelection(event) {
        const selectedFiles = Array.from(event.target.files);
        this.addFolderFiles(selectedFiles);
    }

    handleDragOver(event) {
        event.preventDefault();
        this.uploadArea.classList.add('dragover');
    }

    handleDragLeave(event) {
        event.preventDefault();
        this.uploadArea.classList.remove('dragover');
    }

    handleDrop(event) {
        event.preventDefault();
        this.uploadArea.classList.remove('dragover');
        
        const droppedFiles = Array.from(event.dataTransfer.files);
        this.addFiles(droppedFiles);
    }

    addFiles(newFiles) {
        const supportedTypes = ['.txt', '.pdf', '.doc', '.docx', '.html', '.htm'];
        const validFiles = newFiles.filter(file => {
            const extension = '.' + file.name.split('.').pop().toLowerCase();
            return supportedTypes.includes(extension);
        });

        validFiles.forEach(file => {
            const relativePath = file.webkitRelativePath || file.name;
            const isDuplicate = this.files.some(existingFile => 
                (existingFile.webkitRelativePath || existingFile.name) === relativePath
            );
            
            if (!isDuplicate) {
                this.files.push(file);
            }
        });

        this.updateFileList();
        this.updateProcessButton();

        const invalidCount = newFiles.length - validFiles.length;
        if (invalidCount > 0) {
            NotificationManager.show(`${invalidCount} files were skipped (unsupported format)`, 'warning');
        }

        if (validFiles.length > 0) {
            NotificationManager.show(`Added ${validFiles.length} files`, 'success');
        }
    }

    addFolderFiles(folderFiles) {
        const processedFiles = this.fileTreeProcessor.scanFiles(folderFiles);
        
        processedFiles.forEach(fileInfo => {
            const isDuplicate = this.files.some(existingFile => 
                (existingFile.webkitRelativePath || existingFile.name) === fileInfo.relativePath
            );
            
            if (!isDuplicate) {
                this.files.push(fileInfo.file);
            }
        });

        this.updateFileList();
        this.updateProcessButton();

        const skippedCount = folderFiles.length - processedFiles.length;
        if (skippedCount > 0) {
            NotificationManager.show(`${skippedCount} files were skipped (unsupported format)`, 'warning');
        }

        if (processedFiles.length > 0) {
            const folderName = processedFiles[0].relativePath.split('/')[0] || 'folder';
            NotificationManager.show(`Added ${processedFiles.length} files from "${folderName}"`, 'success');
        }
    }

    updateFileList() {
        if (this.files.length === 0) {
            this.fileList.innerHTML = '';
            return;
        }

        const fileListHTML = `
            <div style="background: var(--gray-50); border-radius: 10px; padding: 20px; margin-top: 20px;">
                <h4 style="margin-bottom: 15px; color: var(--gray-800);">
                    <i class="fas fa-list"></i> Selected Files (${this.files.length})
                    ${this.currentMode === 'folder' ? '<span style="margin-left: 10px; font-size: 0.8rem; color: var(--gray-600);">📁 Folder Mode</span>' : ''}
                </h4>
                <div style="max-height: 250px; overflow-y: auto;">
                    ${this.files.map((file, index) => {
                        const relativePath = file.webkitRelativePath || file.name;
                        const folderDepth = relativePath.split('/').length - 1;
                        const indent = folderDepth > 0 ? `margin-left: ${folderDepth * 20}px;` : '';
                        const pathDisplay = folderDepth > 0 ? 
                            `<div style="font-size: 0.8rem; color: var(--gray-500); ${indent}">${relativePath}</div>` : '';
                        
                        return `
                            <div style="padding: 8px 0; border-bottom: 1px solid var(--gray-200);">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div style="flex: 1;">
                                        <div style="${indent}">
                                            <i class="fas fa-file-medical" style="color: var(--primary-color); margin-right: 8px;"></i>
                                            <span style="font-weight: 500;">${file.name}</span>
                                            <span style="color: var(--gray-600); margin-left: 10px;">(${this.formatFileSize(file.size)})</span>
                                        </div>
                                        ${pathDisplay}
                                    </div>
                                    <button onclick="processor.removeFile(${index})" 
                                            style="background: var(--danger-color); color: white; border: none; border-radius: 5px; padding: 4px 8px; cursor: pointer; font-size: 0.8rem;">
                                        <i class="fas fa-times"></i>
                                    </button>
                                </div>
                            </div>
                        `;
                    }).join('')}
                </div>
            </div>
        `;

        this.fileList.innerHTML = fileListHTML;
    }

    removeFile(index) {
        this.files.splice(index, 1);
        this.updateFileList();
        this.updateProcessButton();
    }

    updateProcessButton() {
        this.processBtn.disabled = this.files.length === 0 || this.isProcessing;
    }

    async startProcessing() {
        if (this.files.length === 0 || this.isProcessing) return;

        this.isProcessing = true;
        this.results = [];
        this.currentFileIndex = 0;
        this.completedFiles = 0;

        this.resultsSection.style.display = 'block';
        this.progressContainer.style.display = 'block';

        this.updateProgress(0, this.files.length);
        this.progressBar.style.width = '0%';
        this.currentProcessing.innerHTML = `
            <div class="processing-file">
                <div class="spinner"></div>
                <span>Initializing...</span>
            </div>
        `;

        // Clear previous results
        this.resultsTableBody.innerHTML = '';

        // Create placeholder rows
        this.files.forEach((file, index) => {
            this.createPlaceholderRow(file, index);
        });

        // Process files in parallel with concurrency limit
        const concurrencyLimit = 3;
        const batches = [];
        for (let i = 0; i < this.files.length; i += concurrencyLimit) {
            batches.push(this.files.slice(i, i + concurrencyLimit));
        }

        for (const batch of batches) {
            const promises = batch.map((file, batchIndex) => {
                const globalIndex = batches.indexOf(batch) * concurrencyLimit + batchIndex;
                return this.processFile(file, globalIndex);
            });
            
            await Promise.all(promises);
        }

        this.isProcessing = false;
        this.updateProcessButton();
        NotificationManager.show('All files processed!', 'success');
    }

    createPlaceholderRow(file, index) {
        const fullPath = file.webkitRelativePath || file.name;
        const row = document.createElement('tr');
        row.id = `result-row-${index}`;
        row.className = 'animate__animated animate__fadeInUp';
        row.innerHTML = TableRenderer.createPlaceholderRow(fullPath);
        this.resultsTableBody.appendChild(row);
    }

    async processFile(file, index) {
        // CRITICAL: Separate full path for display vs filename for backend
        const fullPath = file.webkitRelativePath || file.name;  // Full path for frontend display
        const filename = file.name; // Only filename for backend title processing
        
        // Update row to processing state
        const row = document.getElementById(`result-row-${index}`);
        if (row) {
            row.innerHTML = TableRenderer.createProcessingRow(fullPath);
        }

        this.showCurrentProcessing(fullPath);

        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch('/process-spreadsheet', {
                method: 'POST',
                body: formData
            });

            const rawResult = await response.json();
            
            // **CRITICAL FIX: Override filepath with full path from frontend**
            const cleanResult = DataProcessor.cleanAndMapResult(rawResult, fullPath);

            if (response.ok) {
                this.updateResultRow(index, cleanResult, 'success');
            } else {
                cleanResult.status = 'error';
                cleanResult.error = rawResult.detail || 'Processing failed';
                this.updateResultRow(index, cleanResult, 'error');
            }
            
            this.results.push(cleanResult);

        } catch (error) {
            const errorResult = DataProcessor.cleanAndMapResult({
                title: 'Client-side Error',
                status: 'error',
                error: error.message
            }, fullPath);
            
            this.updateResultRow(index, errorResult, 'error');
            this.results.push(errorResult);
        } finally {
            this.completedFiles++;
            this.updateProgress(this.completedFiles, this.files.length);
            this.progressBar.style.width = `${(this.completedFiles / this.files.length) * 100}%`;
            this.updateStats();
        }
    }

    updateResultRow(index, result, type) {
        const row = document.getElementById(`result-row-${index}`);
        if (!row) return;

        const isSuccess = type === 'success';
        row.innerHTML = TableRenderer.createTableRow(result, isSuccess);
    }

    updateStats() {
        const successful = this.results.filter(r => r.status === 'completed').length;
        const errors = this.results.filter(r => r.status === 'error').length;
        const totalCodes = this.results.reduce((sum, r) => {
            if (r.icd_code_hierarchy) {
                return sum + r.icd_code_hierarchy.split(',').filter(code => code.trim()).length;
            }
            return sum;
        }, 0);

        this.totalFiles.textContent = this.files.length;
        this.successCount.textContent = successful;
        this.errorCount.textContent = errors;
        this.totalCodes.textContent = totalCodes;
    }

    showCurrentProcessing(filename) {
        this.currentProcessing.innerHTML = `
            <div class="processing-file">
                <div class="spinner"></div>
                <span>Processing: ${filename}</span>
            </div>
        `;
    }

    updateProgress(current, total) {
        this.progressStats.textContent = `${current} / ${total} completed`;
        if (current === total) {
            this.currentProcessing.innerHTML = `
                <div class="processing-file" style="background-color: #d1fae5; border-left-color: var(--success-color);">
                    <i class="fas fa-check-circle" style="color: var(--success-color);"></i>
                    <span>All files processed!</span>
                </div>
            `;
        }
    }

    clearAll() {
        this.files = [];
        this.results = [];
        this.updateFileList();
        this.updateProcessButton();
        this.resultsSection.style.display = 'none';
        this.progressContainer.style.display = 'none';
        this.updateStats();
        NotificationManager.show('All files cleared', 'info');
    }

    newBatch() {
        this.clearAll();
        this.fileInput.value = '';
    }

    // Export functionality
    showExportModal() {
        if (this.results.length === 0) {
            NotificationManager.show('No results to export', 'warning');
            return;
        }
        this.exportModal.style.display = 'block';
    }

    hideExportModal() {
        this.exportModal.style.display = 'none';
    }

    exportAsCSV() {
        const csvContent = ExportManager.convertToCSV(this.results);
        ExportManager.downloadFile(csvContent, 'medical_coding_results.csv', 'text/csv');
        this.hideExportModal();
        NotificationManager.show('CSV exported successfully', 'success');
    }

    exportAsExcel() {
        ExportManager.exportAsExcel(this.results);
        this.hideExportModal();
        NotificationManager.show('Excel file exported successfully', 'success');
    }

    exportAsJSON() {
        const jsonContent = JSON.stringify(this.results, null, 2);
        ExportManager.downloadFile(jsonContent, 'medical_coding_results.json', 'application/json');
        this.hideExportModal();
        NotificationManager.show('JSON exported successfully', 'success');
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Initialize the processor when the DOM is loaded
let processor;
document.addEventListener('DOMContentLoaded', function() {
    processor = new SpreadsheetProcessor();
}); 