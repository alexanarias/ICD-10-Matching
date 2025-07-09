// AI Medical Coding - Spreadsheet Interface JavaScript - Modular & Clean

// ZIP File Processor for zip uploads
class ZipFileProcessor {
    constructor() {
        this.supportedTypes = ['.txt', '.pdf', '.doc', '.docx', '.html', '.htm'];
        this.maxZipSize = 100 * 1024 * 1024; // 100MB limit
    }

    async extractZipFiles(zipFile) {
        // Validate ZIP file size
        if (zipFile.size > this.maxZipSize) {
            throw new Error(`ZIP file too large (max ${Math.round(this.maxZipSize / 1024 / 1024)}MB)`);
        }

        // Validate file type
        if (!zipFile.name.toLowerCase().endsWith('.zip')) {
            throw new Error('Please select a valid ZIP file');
        }

        try {
            // Initialize JSZip
            if (typeof JSZip === 'undefined') {
                throw new Error('ZIP processing library not loaded');
            }

            const zip = new JSZip();
            const contents = await zip.loadAsync(zipFile);
            const extractedFiles = [];
            const skippedFiles = [];

            // Process each file in the ZIP - Handle nested folders properly
            for (let [filename, fileObj] of Object.entries(contents.files)) {
                // Skip directories but NOT files inside directories
                if (fileObj.dir) {
                    continue;
                }
                
                // Skip system files but allow nested files
                if (filename.includes('__MACOSX/') || filename.endsWith('.DS_Store') || 
                    filename.includes('/.') || filename.startsWith('.')) {
                    skippedFiles.push(filename);
                    continue;
                }

                // Extract file extension from the actual filename (not path)
                const actualFilename = this.getCleanFilename(filename);
                const extension = '.' + actualFilename.split('.').pop().toLowerCase();
                
                console.log(`Processing: ${filename} -> Extension: ${extension}`);
                
                if (this.supportedTypes.includes(extension)) {
                    try {
                        // Extract file content as blob
                        const blob = await fileObj.async('blob');
                        
                        // Validate blob has content
                        if (blob.size === 0) {
                            console.warn(`Empty file skipped: ${filename}`);
                            skippedFiles.push(filename);
                            continue;
                        }
                        
                        // Create File object with proper metadata
                        const file = new File([blob], actualFilename, {
                            type: this.getMimeType(extension),
                            lastModified: fileObj.date ? fileObj.date.getTime() : Date.now()
                        });
                        
                        // Add FULL relative path for display purposes (shows folder structure)
                        // Use custom property since webkitRelativePath is read-only
                        Object.defineProperty(file, 'zipPath', {
                            value: filename,
                            writable: false,
                            enumerable: false,
                            configurable: false
                        });
                        
                        // For compatibility with existing folder upload code, try to set webkitRelativePath
                        // This will work for manual file selection but not for programmatically created files
                        try {
                            Object.defineProperty(file, 'webkitRelativePath', {
                                value: filename,
                                writable: false,
                                enumerable: false,
                                configurable: false
                            });
                        } catch (e) {
                            // Fallback: use zipPath for path information
                            console.log('Using zipPath instead of webkitRelativePath for:', filename);
                        }
                        
                        extractedFiles.push(file);
                        
                        console.log(`✅ Extracted: ${filename} (${blob.size} bytes)`);
                    } catch (fileError) {
                        console.warn(`Failed to extract ${filename}:`, fileError);
                        skippedFiles.push(filename);
                    }
                } else {
                    console.log(`❌ Unsupported type: ${filename} (${extension})`);
                    skippedFiles.push(filename);
                }
            }

            return {
                extractedFiles,
                skippedFiles,
                totalFiles: Object.keys(contents.files).filter(name => !contents.files[name].dir).length
            };

        } catch (error) {
            if (error.message.includes('Corrupted zip')) {
                throw new Error('ZIP file appears to be corrupted or invalid');
            }
            throw new Error(`Failed to extract ZIP file: ${error.message}`);
        }
    }

    getCleanFilename(path) {
        // Extract just the filename from the path, handle both / and \ separators
        const parts = path.replace(/\\/g, '/').split('/');
        const filename = parts[parts.length - 1];
        
        // Return the filename, ensuring it's not empty
        return filename || path;
    }

    // Helper function to get file path from either webkitRelativePath or zipPath
    getFilePath(file) {
        return file.webkitRelativePath || file.zipPath || file.name;
    }

    getMimeType(extension) {
        const mimeTypes = {
            '.pdf': 'application/pdf',
            '.doc': 'application/msword',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.html': 'text/html',
            '.htm': 'text/html',
            '.txt': 'text/plain'
        };
        return mimeTypes[extension] || 'application/octet-stream';
    }
}

// File Tree Processor for folder uploads
class FileTreeProcessor {
    constructor() {
        this.supportedTypes = ['.txt', '.pdf', '.doc', '.docx', '.html', '.htm'];
    }

    scanFiles(fileList) {
        const processedFiles = [];
        
        for (const file of fileList) {
            const extension = '.' + file.name.split('.').pop().toLowerCase();
            
            if (this.supportedTypes.includes(extension)) {
                            const filePath = file.webkitRelativePath || file.zipPath || file.name;
            const fileInfo = {
                file: file,
                name: file.name,
                size: file.size,
                relativePath: filePath,
                folderDepth: this.calculateFolderDepth(filePath),
                parentFolder: this.extractParentFolder(filePath)
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
        const filePath = file.webkitRelativePath || file.zipPath || file.name;
        if (filePath && filePath !== file.name) {
            // For folder/zip uploads, remove the root folder from the path
            const pathParts = filePath.split('/');
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
            cpt_codes: rawResult.cpt_codes || '',
            // 🎯 CRITICAL: Include structured data for enhanced exports
            icd_codes_structured: rawResult.icd_codes_structured || []
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

    // 🏷️ ICD Chapter Mapping
    static getICDChapter(rootCode) {
        const chapters = {
            'A': 'Infectious Diseases', 'B': 'Infectious Diseases',
            'C': 'Neoplasms', 'D0': 'Neoplasms', 'D1': 'Neoplasms', 'D2': 'Neoplasms', 'D3': 'Neoplasms', 'D4': 'Neoplasms',
            'D5': 'Blood Disorders', 'D6': 'Blood Disorders', 'D7': 'Blood Disorders', 'D8': 'Blood Disorders',
            'E': 'Endocrine Disorders',
            'F': 'Mental Disorders',
            'G': 'Nervous System',
            'H0': 'Eye Disorders', 'H1': 'Eye Disorders', 'H2': 'Eye Disorders', 'H5': 'Eye Disorders',
            'H6': 'Ear Disorders', 'H9': 'Ear Disorders',
            'I': 'Circulatory System',
            'J': 'Respiratory System',
            'K': 'Digestive System',
            'L': 'Skin Disorders',
            'M': 'Musculoskeletal System',
            'N': 'Genitourinary System',
            'O': 'Pregnancy/Childbirth',
            'P': 'Perinatal Conditions',
            'Q': 'Congenital Malformations',
            'R': 'Symptoms/Signs',
            'S': 'Injuries', 'T': 'Injuries',
            'V': 'External Causes', 'W': 'External Causes', 'X': 'External Causes', 'Y': 'External Causes',
            'Z': 'Health Services'
        };
        
        return chapters[rootCode] || chapters[rootCode.substring(0, 2)] || chapters[rootCode.substring(0, 1)] || 'Other';
    }

    // 🎯 Confidence Level Classification
    static getConfidenceLevel(score) {
        if (score >= 90) return 'High';
        if (score >= 70) return 'Medium';
        return 'Low';
    }

    // 📊 ANALYTICS GENERATOR - Creates business intelligence data
    static generateAnalytics(allCodes) {
        const analytics = {
            rootCodeStats: [],
            confidenceDistribution: [],
            topSpecificCodes: [],
            documentSummary: []
        };

        // Root Code Statistics
        const rootStats = {};
        const documentsByRoot = {};
        
        allCodes.forEach(code => {
            if (!rootStats[code.root_code]) {
                rootStats[code.root_code] = {
                    root_code: code.root_code,
                    category_name: code.code_chapter,
                    code_count: 0,
                    total_confidence: 0,
                    documents: new Set()
                };
            }
            
            rootStats[code.root_code].code_count++;
            rootStats[code.root_code].total_confidence += code.confidence_numeric;
            rootStats[code.root_code].documents.add(code.document_path);
        });

        // Convert to array and calculate averages
        Object.values(rootStats).forEach(stat => {
            analytics.rootCodeStats.push({
                root_code: stat.root_code,
                category_name: stat.category_name,
                document_count: stat.documents.size,
                code_count: stat.code_count,
                avg_confidence: Math.round(stat.total_confidence / stat.code_count) + '%'
            });
        });

        // Confidence Distribution
        const confBuckets = { high: 0, medium: 0, low: 0 };
        allCodes.forEach(code => {
            const level = code.confidence_level.toLowerCase();
            confBuckets[level] = (confBuckets[level] || 0) + 1;
        });

        const total = allCodes.length;
        analytics.confidenceDistribution = [
            { confidence_range: '90-100% (High)', code_count: confBuckets.high, percentage: Math.round((confBuckets.high / total) * 100) + '%' },
            { confidence_range: '70-89% (Medium)', code_count: confBuckets.medium, percentage: Math.round((confBuckets.medium / total) * 100) + '%' },
            { confidence_range: 'Below 70% (Low)', code_count: confBuckets.low, percentage: Math.round((confBuckets.low / total) * 100) + '%' }
        ];

        // Top Specific Codes
        const codeFreq = {};
        allCodes.forEach(code => {
            if (!codeFreq[code.icd_code]) {
                codeFreq[code.icd_code] = {
                    icd_code: code.icd_code,
                    description: code.enhanced_description.substring(0, 60) + '...',
                    frequency: 0,
                    total_confidence: 0
                };
            }
            codeFreq[code.icd_code].frequency++;
            codeFreq[code.icd_code].total_confidence += code.confidence_numeric;
        });

        analytics.topSpecificCodes = Object.values(codeFreq)
            .map(code => ({
                ...code,
                avg_confidence: Math.round(code.total_confidence / code.frequency) + '%'
            }))
            .sort((a, b) => b.frequency - a.frequency)
            .slice(0, 15); // Top 15

        return analytics;
    }

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
        console.log('🚀 Starting Multi-Sheet Export...');
        
        // 🔍 DEBUG: Let's see what we're actually getting
        console.log('🔍 DEBUGGING DATA STRUCTURE:');
        console.log('Total rows:', data.length);
        if (data.length > 0) {
            console.log('First row keys:', Object.keys(data[0]));
            console.log('First row icd_codes_structured:', data[0].icd_codes_structured);
            console.log('Type of icd_codes_structured:', typeof data[0].icd_codes_structured);
            if (data[0].icd_codes_structured) {
                console.log('Length of structured codes:', data[0].icd_codes_structured.length);
            }
        }
        
        // 🎯 STEP 1: Extract individual codes from structured data
        const allCodes = [];
        data.forEach(row => {
            if (row.icd_codes_structured && row.icd_codes_structured.length > 0) {
                // Use structured data (perfect quality)
                row.icd_codes_structured.forEach(code => {
                    allCodes.push({
                        document_path: row.filepath,
                        document_title: row.title,
                        document_gender: row.gender,
                        document_keywords: row.keywords,
                        icd_code: code.icd_code,
                        root_code: code.root_code,
                        code_chapter: ExportManager.getICDChapter(code.root_code),
                        enhanced_description: code.enhanced_description,
                        confidence_score: code.confidence_percentage,
                        confidence_numeric: Math.round(code.confidence_score * 100),
                        confidence_level: ExportManager.getConfidenceLevel(Math.round(code.confidence_score * 100)),
                        document_language: row.language,
                        document_status: row.status
                    });
                });
            }
        });
        
        console.log(`✨ Extracted ${allCodes.length} individual ICD codes from ${data.length} documents`);
        
        // 🎯 STEP 2: Generate Analytics
        const analytics = ExportManager.generateAnalytics(allCodes);
        console.log('📊 Generated analytics insights');
        
        // 🎯 STEP 3: Create the  Multi-Sheet Workbook
        const workbook = XLSX.utils.book_new();
        
        // 🔥 SHEET 1: Summary (Original Format - Backward Compatibility)
        console.log('📋 Creating Summary sheet...');
        const summarySheet = XLSX.utils.json_to_sheet(data.map(row => ({
            'FilePath': row.filepath,
            'Title': row.title,
            'Gender': row.gender,
            'Unique Name': row.unique_name,
            'Keywords': row.keywords,
            'ICD Code Root': row.icd_code_root,
            'ICD Code Hierarchy': row.icd_code_hierarchy,
            'Details - Description': row.details_description,
            'Details - Score': row.details_score,
            'Language': row.language,
            'Status': row.status
        })));
        
        // Summary sheet column widths
        summarySheet['!cols'] = [
            { wch: 35 }, { wch: 30 }, { wch: 8 }, { wch: 25 }, { wch: 40 },
            { wch: 12 }, { wch: 25 }, { wch: 50 }, { wch: 20 }, { wch: 10 }, { wch: 12 }
        ];
        
        XLSX.utils.book_append_sheet(workbook, summarySheet, 'Summary');
        
        // 🚀 SHEET 2: ICD Details (Normalized )
        console.log('📊 Creating ICD Details sheet...');
        const detailsSheet = XLSX.utils.json_to_sheet(allCodes.map(code => ({
            'Document Path': code.document_path,
            'Document Title': code.document_title,
            'ICD Code': code.icd_code,
            'Root Code': code.root_code,
            'Code Chapter': code.code_chapter,
            'Enhanced Description': code.enhanced_description,
            'Confidence Score': code.confidence_score,
            'Confidence Level': code.confidence_level,
            'Document Gender': code.document_gender,
            'Document Keywords': code.document_keywords ? code.document_keywords.substring(0, 100) + '...' : '',
            'Language': code.document_language,
            'Status': code.document_status
        })));
        
        // Details sheet column widths
        detailsSheet['!cols'] = [
            { wch: 30 }, { wch: 25 }, { wch: 10 }, { wch: 8 }, { wch: 18 },
            { wch: 45 }, { wch: 12 }, { wch: 12 }, { wch: 8 }, { wch: 30 }, { wch: 10 }, { wch: 10 }
        ];
        
        // 🌈 CONFIDENCE-BASED COLOR CODING
        ExportManager.applyConfidenceColors(detailsSheet, allCodes);
        
        XLSX.utils.book_append_sheet(workbook, detailsSheet, 'ICD_Details');
        
        // 📈 SHEET 3: Analytics
        console.log('📈 Creating Analytics sheet...');
        
        // Create analytics workbook sections
        const analyticsData = [
            // Header
            { 'Metric': '📊 ICD CODING ANALYTICS DASHBOARD', 'Value': '', 'Details': '' },
            { 'Metric': '', 'Value': '', 'Details': '' },
            
            // Root Code Statistics
            { 'Metric': '🏷️ ROOT CODE STATISTICS', 'Value': '', 'Details': '' },
            { 'Metric': 'Root Code', 'Value': 'Category', 'Details': 'Documents | Codes | Avg Confidence' },
            ...analytics.rootCodeStats.map(stat => ({
                'Metric': stat.root_code,
                'Value': stat.category_name,
                'Details': `${stat.document_count} docs | ${stat.code_count} codes | ${stat.avg_confidence}`
            })),
            
            { 'Metric': '', 'Value': '', 'Details': '' },
            
            // Confidence Distribution
            { 'Metric': '🎯 CONFIDENCE DISTRIBUTION', 'Value': '', 'Details': '' },
            { 'Metric': 'Confidence Range', 'Value': 'Code Count', 'Details': 'Percentage' },
            ...analytics.confidenceDistribution.map(dist => ({
                'Metric': dist.confidence_range,
                'Value': dist.code_count.toString(),
                'Details': dist.percentage
            })),
            
            { 'Metric': '', 'Value': '', 'Details': '' },
            
            // Top Specific Codes
            { 'Metric': '🏆 TOP SPECIFIC CODES', 'Value': '', 'Details': '' },
            { 'Metric': 'ICD Code', 'Value': 'Description', 'Details': 'Frequency | Avg Confidence' },
            ...analytics.topSpecificCodes.slice(0, 10).map(code => ({
                'Metric': code.icd_code,
                'Value': code.description,
                'Details': `${code.frequency}x | ${code.avg_confidence}`
            }))
        ];
        
        const analyticsSheet = XLSX.utils.json_to_sheet(analyticsData);
        
        // Analytics sheet styling
        analyticsSheet['!cols'] = [
            { wch: 20 }, { wch: 35 }, { wch: 25 }
        ];
        
        XLSX.utils.book_append_sheet(workbook, analyticsSheet, 'Analytics');
        
        // 🎯 Enhanced metadata
        workbook.Props = {
            Title: 'ICD-10 Medical Coding Analysis',
            Subject: 'Multi-Sheet Medical Coding Results',
            Author: 'AI Medical Coding System v2.0',
            CreatedDate: new Date(),
            Company: 'Medical AI Analytics'
        };
        
        console.log('✨ Multi-sheet workbook created successfully!');
        console.log(`📊 Summary: ${data.length} documents`);
        console.log(`🔍 Details: ${allCodes.length} individual codes`);
        console.log(`📈 Analytics: ${analytics.rootCodeStats.length} code categories`);
        
        // 🚀 DOWNLOAD
        const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
        XLSX.writeFile(workbook, `ICD_Medical_Coding_Analysis_${timestamp}.xlsx`);
        
        // Show success notification
        NotificationManager.show(`🎉 Multi-sheet Excel exported! ${data.length} docs, ${allCodes.length} codes analyzed`, 'success');
    }

    // 🌈 CONFIDENCE COLOR CODING 
    static applyConfidenceColors(worksheet, codes) {
        const range = XLSX.utils.decode_range(worksheet['!ref']);
        
        for (let row = 1; row <= range.e.r; row++) { // Skip header row
            const codeIndex = row - 1;
            if (codeIndex < codes.length) {
                const confidence = codes[codeIndex].confidence_numeric;
                const confidenceCell = XLSX.utils.encode_cell({ r: row, c: 6 }); // Confidence Score column
                const levelCell = XLSX.utils.encode_cell({ r: row, c: 7 }); // Confidence Level column
                
                let fillColor = '';
                if (confidence >= 90) {
                    fillColor = 'FFD4EDDA'; // Green
                } else if (confidence >= 70) {
                    fillColor = 'FFFFF3CD'; // Yellow
                } else {
                    fillColor = 'FFF8D7DA'; // Red
                }
                
                // Apply background colors
                if (worksheet[confidenceCell]) {
                    worksheet[confidenceCell].s = { fill: { fgColor: { rgb: fillColor } } };
                }
                if (worksheet[levelCell]) {
                    worksheet[levelCell].s = { fill: { fgColor: { rgb: fillColor } } };
                }
            }
        }
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
        this.currentMode = 'files'; // 'files', 'folder', or 'zip'
        
        // Initialize processors
        this.fileTreeProcessor = new FileTreeProcessor();
        this.zipProcessor = new ZipFileProcessor();
        
        this.initializeElements();
        this.bindEvents();
    }

    initializeElements() {
        // Main elements
        this.fileInput = document.getElementById('fileInput');
        this.folderInput = document.getElementById('folderInput');
        this.zipInput = document.getElementById('zipInput');
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
        this.zipInput.addEventListener('change', (e) => this.handleZipSelection(e));
        
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
        
        // Update UI elements based on mode
        if (mode === 'folder') {
            this.fileInput.style.display = 'none';
            this.folderInput.style.display = 'block';
            this.zipInput.style.display = 'none';
            this.uploadIcon.className = 'fas fa-folder-open';
            this.uploadText.textContent = 'Drag & Drop Folder Here';
            this.uploadHint.innerHTML = `
                or click to browse<br>
                <small>Supports: PDF, DOC, DOCX, TXT, HTML | Entire folder structure</small><br>
                <strong style="color: var(--primary-color);">Will process all supported files recursively</strong>
            `;
            this.selectFilesBtn.innerHTML = '<i class="fas fa-folder"></i> Select Folder';
        } else if (mode === 'zip') {
            this.fileInput.style.display = 'none';
            this.folderInput.style.display = 'none';
            this.zipInput.style.display = 'block';
            this.uploadIcon.className = 'fas fa-file-archive';
            this.uploadText.textContent = 'Upload ZIP File';
            this.uploadHint.innerHTML = `
                or click to browse<br>
                <small>Supports: ZIP files containing PDF, DOC, DOCX, TXT, HTML</small><br>
                <strong style="color: var(--primary-color);">Will extract and process all supported files</strong>
            `;
            this.selectFilesBtn.innerHTML = '<i class="fas fa-file-archive"></i> Select ZIP';
        } else {
            this.fileInput.style.display = 'block';
            this.folderInput.style.display = 'none';
            this.zipInput.style.display = 'none';
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
        } else if (this.currentMode === 'zip') {
            this.zipInput.click();
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

    async handleZipSelection(event) {
        const zipFile = event.target.files[0];
        if (!zipFile) return;

        try {
            NotificationManager.show('Extracting ZIP file...', 'info');
            
            const result = await this.zipProcessor.extractZipFiles(zipFile);
            const { extractedFiles, skippedFiles, totalFiles } = result;

            if (extractedFiles.length === 0) {
                NotificationManager.show('No supported files found in ZIP', 'warning');
                return;
            }

            // Add extracted files to the file list
            this.addFiles(extractedFiles);

            // Show detailed extraction summary
            let message = `Extracted ${extractedFiles.length} files from "${zipFile.name}"`;
            if (skippedFiles.length > 0) {
                message += ` (${skippedFiles.length} files skipped)`;
            }
            
            // Show folder structure info if nested files found
            const nestedFiles = extractedFiles.filter(f => (f.webkitRelativePath || f.zipPath || '').includes('/'));
            if (nestedFiles.length > 0) {
                message += ` - Found files in subdirectories`;
            }
            
            NotificationManager.show(message, 'success');

            // Log details for debugging
            console.log('ZIP extraction complete:', {
                zipFile: zipFile.name,
                totalFiles,
                extractedFiles: extractedFiles.length,
                skippedFiles: skippedFiles.length
            });

        } catch (error) {
            NotificationManager.show(`ZIP extraction failed: ${error.message}`, 'error');
            console.error('ZIP extraction error:', error);
        }
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
        
        // Check if a single ZIP file was dropped
        if (droppedFiles.length === 1 && droppedFiles[0].name.toLowerCase().endsWith('.zip')) {
            // Switch to ZIP mode and process the ZIP file
            this.switchMode('zip');
            const fakeEvent = { target: { files: droppedFiles } };
            this.handleZipSelection(fakeEvent);
        } else {
            // Handle as regular files
            this.addFiles(droppedFiles);
        }
    }

    addFiles(newFiles) {
        const supportedTypes = ['.txt', '.pdf', '.doc', '.docx', '.html', '.htm'];
        const validFiles = newFiles.filter(file => {
            const extension = '.' + file.name.split('.').pop().toLowerCase();
            return supportedTypes.includes(extension);
        });

        validFiles.forEach(file => {
            const relativePath = file.webkitRelativePath || file.zipPath || file.name;
            const isDuplicate = this.files.some(existingFile => 
                (existingFile.webkitRelativePath || existingFile.zipPath || existingFile.name) === relativePath
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
                (existingFile.webkitRelativePath || existingFile.zipPath || existingFile.name) === fileInfo.relativePath
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
                        const relativePath = file.webkitRelativePath || file.zipPath || file.name;
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
        const fullPath = file.webkitRelativePath || file.zipPath || file.name;
        const row = document.createElement('tr');
        row.id = `result-row-${index}`;
        row.className = 'animate__animated animate__fadeInUp';
        row.innerHTML = TableRenderer.createPlaceholderRow(fullPath);
        this.resultsTableBody.appendChild(row);
    }

    async processFile(file, index) {
        // CRITICAL: Separate full path for display vs filename for backend
        const fullPath = file.webkitRelativePath || file.zipPath || file.name;  // Full path for frontend display
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
        this.folderInput.value = '';
        this.zipInput.value = '';
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