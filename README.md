# SEO Pulse - Enhanced SEO Analysis Tool

A comprehensive Streamlit-based application for analyzing Screaming Frog SEO exports and generating professional audit reports with actionable insights.

## Features

### 📊 Enhanced Visualizations
- **Executive Summary Dashboard** with KPI cards and strategic overview
- **SEO Issues Heatmaps** showing patterns by page type (Homepage, Category, Product, Blog, etc.)
- **Performance Comparison Charts** with radar visualizations
- **Professional Gauge Charts** for individual category scores

### 🎯 Actionable Recommendations
- **Prioritized Action Items** (High/Medium/Low priority)
- **Implementation Difficulty Ratings** (Easy/Medium/Hard)
- **Impact Assessments** to help prioritize fixes
- **Direct Links** to Google documentation, Moz guides, and implementation resources

### 📋 Professional Audit Reports
- **Comprehensive SEO Scoring** across Content SEO (40%), Technical SEO (40%), and User Experience (20%)
- **Detailed Audit Tables** with checkmarks and improvement indicators
- **Multiple Export Formats** (Excel with formatting, CSV)
- **Client-Ready Presentations** with professional styling

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run enhanced_seo_analyzer.py
   ```

## Usage

### Data Export from Screaming Frog
1. Open Screaming Frog SEO Spider
2. Crawl your website (enter URL and click Start)
3. Wait for crawl to complete
4. Go to **Bulk Export menu** → **Response Codes** → **All**
5. Select the **Internal HTML tab**
6. Click **Export** and save as CSV or Excel
7. Upload the file using the application's file uploader

### Analysis Features
- **Single Site Analysis**: Upload one file for detailed audit
- **Competitive Analysis**: Upload multiple files to compare performance
- **Page Type Analysis**: Automatic classification and heatmap visualization
- **Actionable Insights**: Get specific recommendations with implementation guidance

## Supported File Formats
- CSV files (.csv)
- Excel files (.xlsx)

## Analysis Categories

### Content SEO (40% weight)
- Meta title optimization (30-60 characters)
- Meta description optimization (120-160 characters)
- H1 tag presence and optimization
- Internal linking structure
- Schema markup detection
- Image alt text analysis

### Technical SEO (40% weight)
- Response time performance
- HTTP status code health
- Page indexability status
- Mobile/desktop speed analysis
- HTTPS implementation
- XML sitemap and robots.txt

### User Experience (20% weight)
- Mobile friendliness assessment
- Core Web Vitals (LCP, CLS)
- Rich search result optimization

## Scoring System
- **90-100**: Excellent (Green)
- **50-89**: Good (Yellow/Orange)
- **0-49**: Poor (Red)

*"All brands should aim for a score above 90"*

## Export Options
- Individual site Excel audits with professional formatting
- Comprehensive comparison reports
- CSV data for further analysis
- Executive summary dashboards

## Requirements
- Python 3.7+
- Streamlit 1.28+
- See requirements.txt for full dependency list

## License
MIT License - feel free to use for commercial and personal projects.

## Support
For issues or questions, please check the documentation or create an issue in the repository.
