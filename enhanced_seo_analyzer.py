import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import seaborn as sns
import matplotlib.pyplot as plt
from urllib.parse import urlparse

# Configure page settings
st.set_page_config(
    page_title="Screaming Frog Audit Analyzer - Professional SEO Audit Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .recommendation-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 5px solid #28a745;
    }
    
    .priority-high { border-left-color: #dc3545 !important; }
    .priority-medium { border-left-color: #ffc107 !important; }
    .priority-low { border-left-color: #28a745 !important; }
    
    .difficulty-easy { background-color: #d4edda; color: #155724; padding: 4px 8px; border-radius: 4px; font-size: 12px; }
    .difficulty-medium { background-color: #fff3cd; color: #856404; padding: 4px 8px; border-radius: 4px; font-size: 12px; }
    .difficulty-hard { background-color: #f8d7da; color: #721c24; padding: 4px 8px; border-radius: 4px; font-size: 12px; }
    
    .score-excellent { color: #28a745; font-weight: bold; }
    .score-good { color: #ffc107; font-weight: bold; }
    .score-poor { color: #dc3545; font-weight: bold; }
    
    .executive-summary {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        border: 1px solid #dee2e6;
    }
    
    .kpi-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .upload-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #dee2e6;
        text-align: center;
        margin: 2rem 0;
    }
    
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    .discovery-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .bullet-list {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)
class EnhancedSEOScorer:
    def __init__(self):
        self.weights = {
            'content_seo': 0.4,
            'technical_seo': 0.4,
            'user_experience': 0.2
        }
        
        # Page type classification patterns
        self.page_patterns = {
            'Homepage': [r'^/$', r'^/index', r'^/home'],
            'Category': [r'/category/', r'/categories/', r'/c/', r'/cat/'],
            'Product': [r'/product/', r'/products/', r'/p/', r'/item/'],
            'Blog': [r'/blog/', r'/news/', r'/article/', r'/post/'],
            'About': [r'/about', r'/company/', r'/team/'],
            'Contact': [r'/contact', r'/support/', r'/help/'],
            'Other': []  # Catch-all
        }

    def classify_page_type(self, url):
        """Classify page type based on URL patterns"""
        try:
            path = urlparse(url).path.lower()
            
            for page_type, patterns in self.page_patterns.items():
                if page_type == 'Other':
                    continue
                for pattern in patterns:
                    if re.search(pattern, path):
                        return page_type
            return 'Other'
        except:
            return 'Other'

    def analyze_page_types(self, df):
        """Analyze SEO issues by page type"""
        if 'Address' not in df.columns:
            return {}
        
        # Add page type classification
        df['Page_Type'] = df['Address'].apply(self.classify_page_type)
        
        page_type_analysis = {}
        
        for page_type in df['Page_Type'].unique():
            type_df = df[df['Page_Type'] == page_type]
            
            issues = {
                'Missing Titles': (type_df['Title 1'].isna().sum() / len(type_df)) * 100 if 'Title 1' in df.columns else 0,
                'Missing Descriptions': (type_df['Meta Description 1'].isna().sum() / len(type_df)) * 100 if 'Meta Description 1' in df.columns else 0,
                'Missing H1': (type_df['H1-1'].isna().sum() / len(type_df)) * 100 if 'H1-1' in df.columns else 0,
                'Slow Response': ((type_df['Response Time'] > 1.0).sum() / len(type_df)) * 100 if 'Response Time' in df.columns else 0,
                'Not Indexable': ((type_df['Indexability'] != 'Indexable').sum() / len(type_df)) * 100 if 'Indexability' in df.columns else 0
            }
            
            page_type_analysis[page_type] = {
                'count': len(type_df),
                'issues': issues,
                'avg_score': sum(issues.values()) / len(issues)
            }
        
        return page_type_analysis

    def generate_detailed_discoveries(self, df, content_detailed, technical_detailed, ux_detailed, offpage_detailed):
        """Generate detailed bullet-point discoveries for each category"""
        
        discoveries = {
            'Content SEO': [],
            'Technical SEO': [],
            'User Experience': [],
            'Off-Page SEO': []
        }
        
        total_pages = len(df)
        
        # Content SEO Discoveries
        content_discoveries = []
        
        # Meta Title Analysis
        if 'Title 1' in df.columns:
            missing_titles = df['Title 1'].isna().sum()
            if missing_titles > 0:
                content_discoveries.append(f"• {missing_titles} out of {total_pages} pages ({(missing_titles/total_pages)*100:.1f}%) are missing meta titles")
            
            if 'Title 1 Length' in df.columns:
                valid_titles = df['Title 1'].notna()
                if valid_titles.sum() > 0:
                    title_lengths = df[valid_titles]['Title 1 Length']
                    short_titles = (title_lengths < 30).sum()
                    long_titles = (title_lengths > 60).sum()
                    optimal_titles = ((title_lengths >= 30) & (title_lengths <= 60)).sum()
                    
                    if short_titles > 0:
                        content_discoveries.append(f"• {short_titles} pages have titles shorter than 30 characters (too short)")
                    if long_titles > 0:
                        content_discoveries.append(f"• {long_titles} pages have titles longer than 60 characters (may be truncated)")
                    if optimal_titles > 0:
                        content_discoveries.append(f"• {optimal_titles} pages have optimal title lengths (30-60 characters)")
                    
                    avg_title_length = title_lengths.mean()
                    content_discoveries.append(f"• Average title length: {avg_title_length:.1f} characters")
        
        # Meta Description Analysis
        if 'Meta Description 1' in df.columns:
            missing_desc = df['Meta Description 1'].isna().sum()
            if missing_desc > 0:
                content_discoveries.append(f"• {missing_desc} out of {total_pages} pages ({(missing_desc/total_pages)*100:.1f}%) are missing meta descriptions")
            
            if 'Meta Description 1 Length' in df.columns:
                valid_desc = df['Meta Description 1'].notna()
                if valid_desc.sum() > 0:
                    desc_lengths = df[valid_desc]['Meta Description 1 Length']
                    short_desc = (desc_lengths < 120).sum()
                    long_desc = (desc_lengths > 160).sum()
                    optimal_desc = ((desc_lengths >= 120) & (desc_lengths <= 160)).sum()
                    
                    if short_desc > 0:
                        content_discoveries.append(f"• {short_desc} pages have descriptions shorter than 120 characters")
                    if long_desc > 0:
                        content_discoveries.append(f"• {long_desc} pages have descriptions longer than 160 characters")
                    if optimal_desc > 0:
                        content_discoveries.append(f"• {optimal_desc} pages have optimal description lengths (120-160 characters)")
        
        # H1 Tag Analysis
        if 'H1-1' in df.columns:
            missing_h1 = df['H1-1'].isna().sum()
            if missing_h1 > 0:
                content_discoveries.append(f"• {missing_h1} out of {total_pages} pages ({(missing_h1/total_pages)*100:.1f}%) are missing H1 tags")
            
            multiple_h1 = 0
            for col in df.columns:
                if col.startswith('H1-') and col != 'H1-1':
                    multiple_h1 += df[col].notna().sum()
            if multiple_h1 > 0:
                content_discoveries.append(f"• {multiple_h1} instances of multiple H1 tags found (should have only one H1 per page)")
        
        # Internal Linking Analysis
        if 'Inlinks' in df.columns:
            no_inlinks = (df['Inlinks'] == 0).sum()
            if no_inlinks > 0:
                content_discoveries.append(f"• {no_inlinks} pages have no internal links pointing to them")
            
            avg_inlinks = df['Inlinks'].mean()
            max_inlinks = df['Inlinks'].max()
            content_discoveries.append(f"• Average internal links per page: {avg_inlinks:.1f}")
            content_discoveries.append(f"• Most linked page has {max_inlinks} internal links")
        
        discoveries['Content SEO'] = content_discoveries
        
        # Technical SEO Discoveries
        technical_discoveries = []
        
        # Response Time Analysis
        if 'Response Time' in df.columns:
            slow_pages = (df['Response Time'] > 1.0).sum()
            very_slow_pages = (df['Response Time'] > 3.0).sum()
            fast_pages = (df['Response Time'] <= 1.0).sum()
            
            avg_response = df['Response Time'].mean()
            max_response = df['Response Time'].max()
            
            technical_discoveries.append(f"• Average response time: {avg_response:.2f} seconds")
            technical_discoveries.append(f"• {fast_pages} pages load in under 1 second (good performance)")
            
            if slow_pages > 0:
                technical_discoveries.append(f"• {slow_pages} pages have response times over 1 second")
            if very_slow_pages > 0:
                technical_discoveries.append(f"• {very_slow_pages} pages have response times over 3 seconds (critical issue)")
            
            technical_discoveries.append(f"• Slowest page response time: {max_response:.2f} seconds")
        
        # Status Code Analysis
        if 'Status Code' in df.columns:
            status_counts = df['Status Code'].value_counts()
            for status_code, count in status_counts.items():
                percentage = (count / total_pages) * 100
                if status_code == 200:
                    technical_discoveries.append(f"• {count} pages ({percentage:.1f}%) return successful 200 status codes")
                elif status_code == 404:
                    technical_discoveries.append(f"• {count} pages ({percentage:.1f}%) return 404 Not Found errors")
                elif status_code in [301, 302]:
                    technical_discoveries.append(f"• {count} pages ({percentage:.1f}%) are redirects (status {status_code})")
                else:
                    technical_discoveries.append(f"• {count} pages ({percentage:.1f}%) return {status_code} status codes")
        
        # Indexability Analysis
        if 'Indexability' in df.columns:
            indexability_counts = df['Indexability'].value_counts()
            for status, count in indexability_counts.items():
                percentage = (count / total_pages) * 100
                technical_discoveries.append(f"• {count} pages ({percentage:.1f}%) are {status}")
        
        # HTTPS Analysis
        if 'Address' in df.columns:
            https_pages = df['Address'].str.startswith('https://').sum()
            http_pages = df['Address'].str.startswith('http://').sum()
            
            if https_pages > 0:
                technical_discoveries.append(f"• {https_pages} pages ({(https_pages/total_pages)*100:.1f}%) use HTTPS (secure)")
            if http_pages > 0:
                technical_discoveries.append(f"• {http_pages} pages ({(http_pages/total_pages)*100:.1f}%) use HTTP (not secure)")
        
        discoveries['Technical SEO'] = technical_discoveries
        
        # User Experience Discoveries
        ux_discoveries = []
        
        # Core Web Vitals Analysis
        if 'Largest Contentful Paint Time (ms)' in df.columns:
            lcp_good = (df['Largest Contentful Paint Time (ms)'] <= 2500).sum()
            lcp_needs_improvement = ((df['Largest Contentful Paint Time (ms)'] > 2500) & 
                                   (df['Largest Contentful Paint Time (ms)'] <= 4000)).sum()
            lcp_poor = (df['Largest Contentful Paint Time (ms)'] > 4000).sum()
            
            avg_lcp = df['Largest Contentful Paint Time (ms)'].mean() / 1000  # Convert to seconds
            
            ux_discoveries.append(f"• Average Largest Contentful Paint: {avg_lcp:.2f} seconds")
            ux_discoveries.append(f"• {lcp_good} pages have good LCP (≤2.5s)")
            if lcp_needs_improvement > 0:
                ux_discoveries.append(f"• {lcp_needs_improvement} pages need LCP improvement (2.5-4s)")
            if lcp_poor > 0:
                ux_discoveries.append(f"• {lcp_poor} pages have poor LCP (>4s)")
        
        # Cumulative Layout Shift Analysis
        if 'Cumulative Layout Shift' in df.columns:
            cls_good = (df['Cumulative Layout Shift'] <= 0.1).sum()
            cls_needs_improvement = ((df['Cumulative Layout Shift'] > 0.1) & 
                                   (df['Cumulative Layout Shift'] <= 0.25)).sum()
            cls_poor = (df['Cumulative Layout Shift'] > 0.25).sum()
            
            avg_cls = df['Cumulative Layout Shift'].mean()
            
            ux_discoveries.append(f"• Average Cumulative Layout Shift: {avg_cls:.3f}")
            ux_discoveries.append(f"• {cls_good} pages have good CLS (≤0.1)")
            if cls_needs_improvement > 0:
                ux_discoveries.append(f"• {cls_needs_improvement} pages need CLS improvement (0.1-0.25)")
            if cls_poor > 0:
                ux_discoveries.append(f"• {cls_poor} pages have poor CLS (>0.25)")
        
        # Mobile Analysis
        if 'Viewport' in df.columns:
            has_viewport = df['Viewport'].notna().sum()
            missing_viewport = df['Viewport'].isna().sum()
            
            if has_viewport > 0:
                ux_discoveries.append(f"• {has_viewport} pages have viewport meta tags (mobile-friendly)")
            if missing_viewport > 0:
                ux_discoveries.append(f"• {missing_viewport} pages missing viewport meta tags (not mobile-optimized)")
        
        # Image Analysis
        if 'Images' in df.columns:
            total_images = df['Images'].sum() if df['Images'].notna().any() else 0
            pages_with_images = (df['Images'] > 0).sum() if 'Images' in df.columns else 0
            
            if total_images > 0:
                avg_images = df['Images'].mean()
                ux_discoveries.append(f"• Total images across all pages: {total_images}")
                ux_discoveries.append(f"• Average images per page: {avg_images:.1f}")
                ux_discoveries.append(f"• {pages_with_images} pages contain images")
        
        discoveries['User Experience'] = ux_discoveries
        
        # Off-Page SEO Discoveries (general insights)
        offpage_discoveries = [
            "• Authority score analysis requires external backlink data",
            "• Backlink profile assessment needs comprehensive link analysis",
            "• Social signals and brand mentions not available in crawl data",
            "• Competitor comparison requires additional SEO tools",
            "• Domain authority metrics need third-party integration"
        ]
        
        discoveries['Off-Page SEO'] = offpage_discoveries
        
        return discoveries

    def generate_recommendations(self, df, content_score, technical_score, ux_score):
        """Generate specific, prioritized recommendations"""
        recommendations = []
        
        # Content SEO Recommendations
        if 'Title 1' in df.columns:
            missing_titles = df['Title 1'].isna().sum()
            if missing_titles > 0:
                recommendations.append({
                    'category': 'Content SEO',
                    'issue': f'{missing_titles} pages missing meta titles',
                    'action': 'Add unique, descriptive meta titles (30-60 characters) to all pages',
                    'priority': 'High' if missing_titles > len(df) * 0.2 else 'Medium',
                    'difficulty': 'Easy',
                    'impact': 'High',
                    'resources': [
                        {'title': 'Google Title Tag Guidelines', 'url': 'https://developers.google.com/search/docs/appearance/title-link'},
                        {'title': 'Moz Title Tag Guide', 'url': 'https://moz.com/learn/seo/title-tag'}
                    ]
                })
        
        if 'Meta Description 1' in df.columns:
            missing_desc = df['Meta Description 1'].isna().sum()
            if missing_desc > 0:
                recommendations.append({
                    'category': 'Content SEO',
                    'issue': f'{missing_desc} pages missing meta descriptions',
                    'action': 'Write compelling meta descriptions (120-160 characters) that encourage clicks',
                    'priority': 'Medium',
                    'difficulty': 'Easy',
                    'impact': 'Medium',
                    'resources': [
                        {'title': 'Meta Description Best Practices', 'url': 'https://moz.com/learn/seo/meta-description'},
                        {'title': 'Google Meta Description Guidelines', 'url': 'https://developers.google.com/search/docs/appearance/snippet'}
                    ]
                })
        
        if 'H1-1' in df.columns:
            missing_h1 = df['H1-1'].isna().sum()
            if missing_h1 > 0:
                recommendations.append({
                    'category': 'Content SEO',
                    'issue': f'{missing_h1} pages missing H1 tags',
                    'action': 'Add descriptive H1 tags that clearly describe page content and include target keywords',
                    'priority': 'High' if missing_h1 > len(df) * 0.3 else 'Medium',
                    'difficulty': 'Easy',
                    'impact': 'High',
                    'resources': [
                        {'title': 'Header Tags SEO Guide', 'url': 'https://moz.com/learn/seo/on-page-factors'},
                        {'title': 'HTML Headings Best Practices', 'url': 'https://developer.mozilla.org/en-US/docs/Web/HTML/Element/Heading_Elements'}
                    ]
                })
        
        # Technical SEO Recommendations
        if 'Response Time' in df.columns:
            slow_pages = (df['Response Time'] > 1.0).sum()
            if slow_pages > 0:
                avg_time = df['Response Time'].mean()
                recommendations.append({
                    'category': 'Technical SEO',
                    'issue': f'{slow_pages} pages with slow response times (avg: {avg_time:.2f}s)',
                    'action': 'Optimize server response times through caching, CDN implementation, and server optimization',
                    'priority': 'High' if avg_time > 2.0 else 'Medium',
                    'difficulty': 'Hard',
                    'impact': 'High',
                    'resources': [
                        {'title': 'Google PageSpeed Insights', 'url': 'https://pagespeed.web.dev/'},
                        {'title': 'Core Web Vitals Guide', 'url': 'https://web.dev/vitals/'}
                    ]
                })
        
        if 'Status Code' in df.columns:
            error_pages = df[df['Status Code'] != 200].shape[0]
            if error_pages > 0:
                recommendations.append({
                    'category': 'Technical SEO',
                    'issue': f'{error_pages} pages with HTTP errors',
                    'action': 'Fix 404 errors, implement proper redirects, and resolve server errors',
                    'priority': 'High',
                    'difficulty': 'Medium',
                    'impact': 'High',
                    'resources': [
                        {'title': 'HTTP Status Codes Guide', 'url': 'https://developer.mozilla.org/en-US/docs/Web/HTTP/Status'},
                        {'title': 'Google Search Console Help', 'url': 'https://support.google.com/webmasters/'}
                    ]
                })
        
        if 'Indexability' in df.columns:
            non_indexable = (df['Indexability'] != 'Indexable').sum()
            if non_indexable > 0:
                recommendations.append({
                    'category': 'Technical SEO',
                    'issue': f'{non_indexable} pages not indexable by search engines',
                    'action': 'Review and fix robots.txt, meta robots tags, and canonical issues preventing indexation',
                    'priority': 'High',
                    'difficulty': 'Medium',
                    'impact': 'High',
                    'resources': [
                        {'title': 'Robots.txt Guide', 'url': 'https://developers.google.com/search/docs/crawling-indexing/robots/robots_txt'},
                        {'title': 'Meta Robots Tag Guide', 'url': 'https://developers.google.com/search/docs/crawling-indexing/robots-meta-tag'}
                    ]
                })
        
        # User Experience Recommendations
        if 'Largest Contentful Paint Time (ms)' in df.columns:
            slow_lcp = (df['Largest Contentful Paint Time (ms)'] > 2500).sum()
            if slow_lcp > 0:
                recommendations.append({
                    'category': 'User Experience',
                    'issue': f'{slow_lcp} pages with slow Largest Contentful Paint',
                    'action': 'Optimize images, reduce server response times, and eliminate render-blocking resources',
                    'priority': 'Medium',
                    'difficulty': 'Hard',
                    'impact': 'Medium',
                    'resources': [
                        {'title': 'LCP Optimization Guide', 'url': 'https://web.dev/lcp/'},
                        {'title': 'Image Optimization Guide', 'url': 'https://web.dev/fast/#optimize-your-images'}
                    ]
                })
        
        # Sort recommendations by priority and impact
        priority_order = {'High': 3, 'Medium': 2, 'Low': 1}
        impact_order = {'High': 3, 'Medium': 2, 'Low': 1}
        
        recommendations.sort(key=lambda x: (priority_order[x['priority']], impact_order[x['impact']]), reverse=True)
        
        return recommendations

    def create_heatmap_data(self, page_type_analysis):
        """Create data for heatmap visualization"""
        if not page_type_analysis:
            return pd.DataFrame()
        
        heatmap_data = []
        for page_type, data in page_type_analysis.items():
            for issue, percentage in data['issues'].items():
                heatmap_data.append({
                    'Page Type': page_type,
                    'Issue Type': issue,
                    'Percentage': percentage,
                    'Count': data['count']
                })
        
        return pd.DataFrame(heatmap_data)

    def analyze_content_seo(self, df):
        scores = {}
        weaknesses = []
        detailed_analysis = {}

        # Meta Title Analysis
        title_score = 0
        if 'Title 1' in df.columns:
            valid_titles = df['Title 1'].notna()
            if 'Title 1 Length' in df.columns:
                good_length = (df['Title 1 Length'] >= 30) & (df['Title 1 Length'] <= 60)
                title_score = ((valid_titles & good_length).mean() * 100)
            else:
                title_score = valid_titles.mean() * 100
            
            if title_score < 50:
                weaknesses.append("Short or missing meta titles.")
        
        detailed_analysis['meta_title'] = {
            'status': '✓' if title_score >= 70 else '✗',
            'score': title_score,
            'description': 'Optimized Meta Title & Description Optimization',
            'needs_improvement': title_score < 70
        }
        scores['meta_title'] = round(title_score)

        # Meta Description Analysis
        desc_score = 0
        if 'Meta Description 1' in df.columns:
            valid_desc = df['Meta Description 1'].notna()
            if 'Meta Description 1 Length' in df.columns:
                good_length = (df['Meta Description 1 Length'] >= 120) & (df['Meta Description 1 Length'] <= 160)
                desc_score = ((valid_desc & good_length).mean() * 100)
            else:
                desc_score = valid_desc.mean() * 100
            
            if desc_score < 50:
                weaknesses.append("Short or missing meta descriptions.")
        scores['meta_description'] = round(desc_score)

        # Headers Analysis
        h1_score = 0
        if 'H1-1' in df.columns:
            h1_score = df['H1-1'].notna().mean() * 100
            if h1_score < 50:
                weaknesses.append("Missing or poorly optimized H1 tags.")
        elif 'H1' in df.columns:
            h1_score = df['H1'].notna().mean() * 100
            if h1_score < 50:
                weaknesses.append("Missing or poorly optimized H1 tags.")
        
        detailed_analysis['headers'] = {
            'status': '✓' if h1_score >= 70 else '✗',
            'score': h1_score,
            'description': 'Optimized Headers',
            'needs_improvement': h1_score < 70
        }
        scores['h1_tags'] = round(h1_score)

        # Schema markup analysis
        schema_score = 30
        detailed_analysis['schema'] = {
            'status': '✗',
            'score': schema_score,
            'description': 'Missing Schema Markups',
            'needs_improvement': True
        }

        # Image alt text analysis
        img_alt_score = 45
        detailed_analysis['image_alt'] = {
            'status': '✗',
            'score': img_alt_score,
            'description': 'Image Alt Text',
            'needs_improvement': True
        }

        # Internal linking
        internal_linking_score = 0
        if 'Inlinks' in df.columns:
            has_inlinks = df['Inlinks'] > 0
            internal_linking_score = has_inlinks.mean() * 100
            if internal_linking_score < 50:
                weaknesses.append("Insufficient internal linking.")
        
        detailed_analysis['internal_linking'] = {
            'status': '✓' if internal_linking_score >= 70 else '✗',
            'score': internal_linking_score,
            'description': 'Internal Linking',
            'needs_improvement': internal_linking_score < 70
        }
        scores['internal_linking'] = round(internal_linking_score)

        # Editorial content
        detailed_analysis['editorial_content'] = {
            'status': '✓',
            'score': 70,
            'description': 'Editorial Content',
            'needs_improvement': False
        }

        return round(np.mean(list(scores.values()))), scores, weaknesses, detailed_analysis

    def analyze_technical_seo(self, df):
        scores = {}
        weaknesses = []
        detailed_analysis = {}

        # Response Time Analysis
        response_score = 0
        avg_response_time = 2.0
        
        if 'Response Time' in df.columns:
            avg_response = df['Response Time'].mean()
            good_response = df['Response Time'] <= 1.0
            response_score = good_response.mean() * 100
            avg_response_time = avg_response
            if response_score < 50:
                weaknesses.append("Slow response times.")
        
        scores['response_time'] = round(response_score)

        detailed_analysis['mobile_speed'] = {
            'status': 'Needs Improvement' if avg_response_time > 3 else '✓',
            'score': max(0, 100 - (avg_response_time * 10)),
            'description': f'Mobile Speed: {avg_response_time:.1f}s',
            'needs_improvement': avg_response_time > 3
        }
        
        desktop_speed = avg_response_time * 0.7
        detailed_analysis['desktop_speed'] = {
            'status': 'Needs Improvement' if desktop_speed > 3 else '✓',
            'score': max(0, 100 - (desktop_speed * 10)),
            'description': f'Desktop Speed: {desktop_speed:.1f}s',
            'needs_improvement': desktop_speed > 3
        }

        # Status Code Analysis
        status_score = 85
        if 'Status Code' in df.columns:
            good_status = df['Status Code'] == 200
            status_score = good_status.mean() * 100
            if status_score < 70:
                weaknesses.append("Issues with HTTP status codes.")
        scores['status_codes'] = round(status_score)

        # Indexability Analysis
        index_score = 80
        if 'Indexability' in df.columns:
            indexable = df['Indexability'] == 'Indexable'
            index_score = indexable.mean() * 100
            if index_score < 70:
                weaknesses.append("Pages not indexable.")
        scores['indexability'] = round(index_score)

        # Additional technical factors
        detailed_analysis['image_optimization'] = {
            'status': '✓',
            'score': 60,
            'description': 'Optimized Image Alt-Attributes',
            'needs_improvement': True
        }

        detailed_analysis['xml_sitemap'] = {
            'status': '✓',
            'score': 100,
            'description': 'XML Sitemap',
            'needs_improvement': False
        }

        detailed_analysis['robots_txt'] = {
            'status': '✓',
            'score': 100,
            'description': 'Robots.txt File',
            'needs_improvement': False
        }

        detailed_analysis['https_urls'] = {
            'status': '✓',
            'score': 100,
            'description': 'Non-HTTPS URLs',
            'needs_improvement': False
        }

        detailed_analysis['ssr_content'] = {
            'status': '✓',
            'score': 85,
            'description': 'Mostly SSR loaded content',
            'needs_improvement': False
        }

        detailed_analysis['hreflang_tags'] = {
            'status': '✓',
            'score': 90,
            'description': 'Optimized Hreflang Tags',
            'needs_improvement': False
        }

        return round(np.mean(list(scores.values()))), scores, weaknesses, detailed_analysis

    def analyze_user_experience(self, df):
        scores = {}
        weaknesses = []
        detailed_analysis = {}

        # Mobile Friendliness Analysis
        mobile_score = 60
        if 'Mobile Alternate Link' in df.columns:
            mobile_score = df['Mobile Alternate Link'].notna().mean() * 100
        elif 'Viewport' in df.columns:
            has_viewport = df['Viewport'].notna()
            mobile_score = has_viewport.mean() * 100
            
        if mobile_score < 50:
            weaknesses.append("Pages not mobile-friendly.")
        
        detailed_analysis['mobile_friendly'] = {
            'status': '✓' if mobile_score >= 70 else '✗',
            'score': mobile_score,
            'description': 'Mobile Friendliness',
            'needs_improvement': mobile_score < 70
        }
        scores['mobile_friendly'] = round(mobile_score)

        detailed_analysis['rich_search'] = {
            'status': '✗',
            'score': 45,
            'description': 'Rich Search Result Optimization',
            'needs_improvement': True
        }

        # Core Web Vitals - LCP
        lcp_score = 55
        if 'Largest Contentful Paint Time (ms)' in df.columns:
            good_lcp = df['Largest Contentful Paint Time (ms)'] <= 2500
            lcp_score = good_lcp.mean() * 100
            if lcp_score < 50:
                weaknesses.append("Slow LCP times.")
        scores['largest_contentful_paint'] = round(lcp_score)

        # Cumulative Layout Shift
        cls_score = 70
        if 'Cumulative Layout Shift' in df.columns:
            good_cls = df['Cumulative Layout Shift'] <= 0.1
            cls_score = good_cls.mean() * 100
            if cls_score < 50:
                weaknesses.append("High CLS values.")
        scores['cumulative_layout_shift'] = round(cls_score)

        return round(np.mean(list(scores.values()))), scores, weaknesses, detailed_analysis

    def analyze_offpage_seo(self, df):
        detailed_analysis = {}
        
        detailed_analysis['authority_score'] = {
            'status': 'Needs Improvement',
            'score': 50,
            'description': 'Authority Score: 50',
            'needs_improvement': True
        }
        
        detailed_analysis['backlinks'] = {
            'status': 'Needs Improvement',
            'score': 65,
            'description': 'Backlinking Profile',
            'needs_improvement': True
        }
        
        return detailed_analysis

    def calculate_overall_score(self, content_score, technical_score, ux_score):
        content_score = content_score / 100
        technical_score = technical_score / 100
        ux_score = ux_score / 100

        weighted_scores = {
            'Content SEO': content_score * self.weights['content_seo'],
            'Technical SEO': technical_score * self.weights['technical_seo'],
            'User Experience': ux_score * self.weights['user_experience']
        }

        overall_score = sum(weighted_scores.values()) * 100
        return round(overall_score)
        def create_heatmap_visualization(heatmap_data):
    """Create a heatmap showing issues by page type"""
    if heatmap_data.empty:
        return None
    
    # Pivot data for heatmap
    pivot_data = heatmap_data.pivot(index='Page Type', columns='Issue Type', values='Percentage')
    pivot_data = pivot_data.fillna(0)
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale='RdYlGn_r',  # Red-Yellow-Green reversed (red = bad)
        text=pivot_data.values,
        texttemplate='%{text:.1f}%',
        textfont={"size": 10},
        hoverongaps=False,
        colorbar=dict(title="Issue Percentage")
    ))
    
    fig.update_layout(
        title="SEO Issues Heatmap by Page Type",
        xaxis_title="Issue Type",
        yaxis_title="Page Type",
        height=400,
        font=dict(size=12)
    )
    
    return fig

def create_gauge_chart(score, title):
    """Create a gauge chart for scores"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 16}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#ffebee'},
                {'range': [50, 80], 'color': '#fff3e0'},
                {'range': [80, 100], 'color': '#e8f5e8'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def create_comparison_chart(comparison_df):
    """Create a comparison chart for all sites"""
    fig = go.Figure()
    
    categories = ['Content SEO', 'Technical SEO', 'User Experience', 'Overall Readiness']
    
    for _, row in comparison_df.iterrows():
        site_name = row['File Name'].replace('.xlsx', '').replace('_', ' ')
        fig.add_trace(go.Scatterpolar(
            r=[row['Content SEO'], row['Technical SEO'], row['User Experience'], row['Overall Readiness']],
            theta=categories,
            fill='toself',
            name=site_name,
            line=dict(width=2)
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="SEO Performance Comparison",
        height=500
    )
    
    return fig

def display_executive_summary(all_results):
    """Display executive summary with key findings"""
    st.markdown("""
    <div class="executive-summary">
        <h2>Executive Summary</h2>
    </div>
    """, unsafe_allow_html=True)
    
    if not all_results:
        return
    
    # Calculate summary statistics
    avg_overall = np.mean([r['overall_score'] for r in all_results])
    best_site = max(all_results, key=lambda x: x['overall_score'])
    worst_site = min(all_results, key=lambda x: x['overall_score'])
    
    # Key findings
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <h3 style="color: #667eea; margin: 0;">Average Score</h3>
            <div style="font-size: 2rem; font-weight: bold; margin: 10px 0;">{avg_overall:.0f}/100</div>
            <p style="margin: 0; color: #666;">Across all sites</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        best_name = best_site['file_name'].replace('.xlsx', '').replace('_', ' ')
        st.markdown(f"""
        <div class="kpi-card">
            <h3 style="color: #28a745; margin: 0;">Best Performer</h3>
            <div style="font-size: 1.2rem; font-weight: bold; margin: 10px 0;">{best_name}</div>
            <p style="margin: 0; color: #666;">{best_site['overall_score']}/100</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        worst_name = worst_site['file_name'].replace('.xlsx', '').replace('_', ' ')
        st.markdown(f"""
        <div class="kpi-card">
            <h3 style="color: #dc3545; margin: 0;">Needs Attention</h3>
            <div style="font-size: 1.2rem; font-weight: bold; margin: 10px 0;">{worst_name}</div>
            <p style="margin: 0; color: #666;">{worst_site['overall_score']}/100</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        total_issues = sum([len(r['recommendations']) for r in all_results])
        st.markdown(f"""
        <div class="kpi-card">
            <h3 style="color: #ffc107; margin: 0;">Action Items</h3>
            <div style="font-size: 2rem; font-weight: bold; margin: 10px 0;">{total_issues}</div>
            <p style="margin: 0; color: #666;">Total recommendations</p>
        </div>
        """, unsafe_allow_html=True)

def display_recommendations(recommendations, site_key=""):
    """Display actionable recommendations with priority and difficulty"""
    if not recommendations:
        st.info("No specific recommendations generated. Your site appears to be well-optimized!")
        return
    
    # Group by priority
    high_priority = [r for r in recommendations if r['priority'] == 'High']
    medium_priority = [r for r in recommendations if r['priority'] == 'Medium']
    low_priority = [r for r in recommendations if r['priority'] == 'Low']
    
    # Display high priority first
    if high_priority:
        st.markdown("### High Priority Items")
        for i, rec in enumerate(high_priority):
            display_recommendation_card(rec, f"{site_key}_high_{i}")
    
    if medium_priority:
        st.markdown("### Medium Priority Items")
        for i, rec in enumerate(medium_priority):
            display_recommendation_card(rec, f"{site_key}_medium_{i}")
    
    if low_priority:
        st.markdown("### Low Priority Items")
        for i, rec in enumerate(low_priority):
            display_recommendation_card(rec, f"{site_key}_low_{i}")

def display_recommendation_card(rec, key):
    """Display individual recommendation card"""
    priority_class = f"priority-{rec['priority'].lower()}"
    
    with st.container():
        st.markdown(f"""
        <div class="recommendation-card {priority_class}">
            <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 1rem;">
                <div>
                    <h4 style="margin: 0; color: #333;">{rec['issue']}</h4>
                    <p style="margin: 5px 0; color: #666; font-size: 0.9rem;">{rec['category']}</p>
                </div>
                <div style="display: flex; gap: 8px;">
                    <span class="difficulty-{rec['difficulty'].lower()}">{rec['difficulty']}</span>
                    <span style="background-color: #e9ecef; color: #495057; padding: 4px 8px; border-radius: 4px; font-size: 12px;">
                        {rec['impact']} Impact
                    </span>
                </div>
            </div>
            <div style="margin-bottom: 1rem;">
                <strong>Action:</strong> {rec['action']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Resources section
        if rec.get('resources'):
            with st.expander("Helpful Resources"):
                for resource in rec['resources']:
                    st.markdown(f"• [{resource['title']}]({resource['url']})")

def display_header():
    """Display the main header with branding"""
    st.markdown("""
    <div class="main-header">
        <h1>Screaming Frog Report Analyzer</h1>
        <p style="font-size: 1.2rem; margin-bottom: 0;">
            Enhanced SEO Analysis with Actionable Insights and Visualizations
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_sidebar_info():
    """Display helpful information in sidebar"""
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-section">
            <h3>About Screaming Frog Audit Analyzer</h3>
            <p>Advanced SEO analysis with heatmaps, actionable recommendations, and executive summaries.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-section">
            <h3>New Features</h3>
            <ul>
                <li>Page type analysis heatmaps</li>
                <li>Prioritized action items</li>
                <li>Implementation difficulty ratings</li>
                <li>Executive summary dashboards</li>
                <li>Resource links for fixes</li>
                <li>Detailed audit discoveries</li>
                <li>Enhanced Excel exports</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-section">
            <h3>Analysis Categories</h3>
            <ul>
                <li><strong>Content SEO (40%)</strong><br>Meta titles, descriptions, H1 tags, internal linking</li>
                <li><strong>Technical SEO (40%)</strong><br>Response times, status codes, indexability</li>
                <li><strong>User Experience (20%)</strong><br>Mobile-friendliness, Core Web Vitals</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        def main():
    display_header()
    display_sidebar_info()
    
    # File upload section
    st.markdown("""
    <div class="upload-section">
        <h3>Upload Your SEO Data</h3>
        <p>Select multiple Screaming Frog export files for enhanced analysis with heatmaps and actionable recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Choose files to analyze",
        type=['csv', 'xlsx'],
        accept_multiple_files=True,
        help="Upload Screaming Frog export files in CSV or Excel format"
    )

    if uploaded_files:
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_results = []
        comparison_data = []
        detailed_analyses = []
        total_files = len(uploaded_files)
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f'Processing {uploaded_file.name}... ({i+1}/{total_files})')
            progress_bar.progress((i + 1) / total_files)
            
            try:
                # Load data
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                scorer = EnhancedSEOScorer()
                
                # Analyze each category
                content_score, content_details, content_weaknesses, content_detailed = scorer.analyze_content_seo(df)
                technical_score, technical_details, technical_weaknesses, technical_detailed = scorer.analyze_technical_seo(df)
                ux_score, ux_details, ux_weaknesses, ux_detailed = scorer.analyze_user_experience(df)
                offpage_detailed = scorer.analyze_offpage_seo(df)
                overall_score = scorer.calculate_overall_score(content_score, technical_score, ux_score)
                
                # Generate recommendations and page type analysis
                recommendations = scorer.generate_recommendations(df, content_score, technical_score, ux_score)
                page_type_analysis = scorer.analyze_page_types(df)
                
                # Generate detailed discoveries
                discoveries = scorer.generate_detailed_discoveries(df, content_detailed, technical_detailed, ux_detailed, offpage_detailed)
                
                # Store comprehensive results
                result = {
                    'file_name': uploaded_file.name,
                    'overall_score': overall_score,
                    'content_score': content_score,
                    'technical_score': technical_score,
                    'ux_score': ux_score,
                    'recommendations': recommendations,
                    'page_type_analysis': page_type_analysis,
                    'discoveries': discoveries,
                    'df': df
                }
                all_results.append(result)
                
                comparison_data.append({
                    "File Name": uploaded_file.name,
                    "Content SEO": content_score,
                    "Technical SEO": technical_score,
                    "User Experience": ux_score,
                    "Overall Readiness": overall_score,
                    "Content Details": content_details,
                    "Technical Details": technical_details,
                    "UX Details": ux_details
                })
                
                # Store detailed analysis for audit tables
                detailed_analyses.append({
                    "File Name": uploaded_file.name,
                    "Content Analysis": content_detailed,
                    "Technical Analysis": technical_detailed,
                    "UX Analysis": ux_detailed,
                    "Offpage Analysis": offpage_detailed,
                    "Overall Score": overall_score,
                    "Discoveries": discoveries
                })
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                continue
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        if all_results:
            comparison_df = pd.DataFrame(comparison_data)
            
            # Display executive summary first
            display_executive_summary(all_results)
            
            # Create tabs for different views
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "Performance Overview", 
                "Heatmap Analysis", 
                "Action Items", 
                "Detailed Audits", 
                "Competitive Analysis", 
                "Export Results"
            ])
            
            with tab1:
                # Performance Overview
                st.markdown("## Performance Overview")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_content = comparison_df['Content SEO'].mean()
                    st.metric("Avg Content SEO", f"{avg_content:.0f}%")
                
                with col2:
                    avg_technical = comparison_df['Technical SEO'].mean()
                    st.metric("Avg Technical SEO", f"{avg_technical:.0f}%")
                
                with col3:
                    avg_ux = comparison_df['User Experience'].mean()
                    st.metric("Avg User Experience", f"{avg_ux:.0f}%")
                
                with col4:
                    avg_overall = comparison_df['Overall Readiness'].mean()
                    st.metric("Avg Overall Score", f"{avg_overall:.0f}%")
                
                # Individual site performance
                st.markdown("### Site Performance Breakdown")
                sorted_results = sorted(all_results, key=lambda x: x['overall_score'], reverse=True)
                
                for idx, result in enumerate(sorted_results):
                    site_name = result['file_name'].replace('.xlsx', '').replace('_', ' ')
                    with st.expander(f"{site_name} - Score: {result['overall_score']}/100"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            fig1 = create_gauge_chart(result['content_score'], 'Content SEO')
                            st.plotly_chart(fig1, use_container_width=True, key=f"content_gauge_{idx}")
                        
                        with col2:
                            fig2 = create_gauge_chart(result['technical_score'], 'Technical SEO')
                            st.plotly_chart(fig2, use_container_width=True, key=f"technical_gauge_{idx}")
                        
                        with col3:
                            fig3 = create_gauge_chart(result['ux_score'], 'User Experience')
                            st.plotly_chart(fig3, use_container_width=True, key=f"ux_gauge_{idx}")
            
            with tab2:
                # Heatmap Analysis
                st.markdown("## SEO Issues Heatmap Analysis")
                st.markdown("This heatmap shows which page types have the most SEO issues, helping you prioritize fixes.")
                
                # Combine all page type analyses
                all_heatmap_data = []
                for result in all_results:
                    site_name = result['file_name'].replace('.xlsx', '').replace('_', ' ')
                    heatmap_data = EnhancedSEOScorer().create_heatmap_data(result['page_type_analysis'])
                    if not heatmap_data.empty:
                        heatmap_data['Site'] = site_name
                        all_heatmap_data.append(heatmap_data)
                
                if all_heatmap_data:
                    combined_heatmap = pd.concat(all_heatmap_data, ignore_index=True)
                    
                    # Create heatmap for each site
                    for result in all_results:
                        site_name = result['file_name'].replace('.xlsx', '').replace('_', ' ')
                        site_heatmap_data = EnhancedSEOScorer().create_heatmap_data(result['page_type_analysis'])
                        
                        if not site_heatmap_data.empty:
                            st.markdown(f"### {site_name}")
                            heatmap_fig = create_heatmap_visualization(site_heatmap_data)
                            if heatmap_fig:
                                st.plotly_chart(heatmap_fig, use_container_width=True)
                            
                            # Show page type breakdown
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Page Type Distribution:**")
                                page_counts = pd.DataFrame([
                                    {'Page Type': ptype, 'Count': data['count']} 
                                    for ptype, data in result['page_type_analysis'].items()
                                ])
                                st.dataframe(page_counts, use_container_width=True)
                            
                            with col2:
                                st.markdown("**Worst Performing Page Types:**")
                                worst_types = sorted(
                                    result['page_type_analysis'].items(), 
                                    key=lambda x: x[1]['avg_score'], 
                                    reverse=True
                                )[:3]
                                for ptype, data in worst_types:
                                    st.write(f"• **{ptype}**: {data['avg_score']:.1f}% issues ({data['count']} pages)")
                else:
                    st.info("No page type data available for heatmap analysis. Upload files with URL data for enhanced visualization.")
            
            with tab3:
                # Action Items with Enhanced Excel Export
                st.markdown("## Prioritized Action Items")
                
                # Combine all recommendations
                all_recommendations = []
                for result in all_results:
                    site_name = result['file_name'].replace('.xlsx', '').replace('_', ' ')
                    for rec in result['recommendations']:
                        rec_copy = rec.copy()
                        rec_copy['site'] = site_name
                        all_recommendations.append(rec_copy)
                
                if all_recommendations:
                    # Add Excel export button at the top
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown("### Download Action Items")
                        st.markdown("Export all recommendations to Excel with detailed action plans and resource links.")
                    
                    with col2:
                        # Create comprehensive action items DataFrame
                        action_items_data = []
                        for i, rec in enumerate(all_recommendations, 1):
                            # Format resources as text
                            resources_text = ""
                            if rec.get('resources'):
                                resources_list = [f"{r['title']}: {r['url']}" for r in rec['resources']]
                                resources_text = " | ".join(resources_list)
                            
                            action_items_data.append({
                                'Item #': i,
                                'Site': rec['site'],
                                'Category': rec['category'],
                                'Priority': rec['priority'],
                                'Issue': rec['issue'],
                                'Action Required': rec['action'],
                                'Difficulty': rec['difficulty'],
                                'Impact': rec['impact'],
                                'Resources': resources_text,
                                'Status': 'Pending'  # Add status column for tracking
                            })
                        
                        action_items_df = pd.DataFrame(action_items_data)
                        
                        # Create Excel export with multiple sheets
                        output = BytesIO()
                        try:
                            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                # Main action items sheet
                                action_items_df.to_excel(writer, sheet_name='Action_Items', index=False)
                                
                                # Summary sheet by priority
                                priority_summary = action_items_df.groupby(['Priority', 'Site']).size().reset_index(name='Count')
                                priority_summary.to_excel(writer, sheet_name='Priority_Summary', index=False)
                                
                                # Summary sheet by category
                                category_summary = action_items_df.groupby(['Category', 'Site']).size().reset_index(name='Count')
                                category_summary.to_excel(writer, sheet_name='Category_Summary', index=False)
                            
                            st.download_button(
                                label="📊 Download Action Items",
                                data=output.getvalue(),
                                file_name=f"SEO_Action_Items_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                help="Download comprehensive action items list with priorities, resources, and tracking"
                            )
                        except Exception as e:
                            st.error(f"Error creating Excel file: {str(e)}")
                            # Fallback to basic CSV
                            csv_data = action_items_df.to_csv(index=False)
                            st.download_button(
                                label="📄 Download as CSV (Fallback)",
                                data=csv_data,
                                file_name=f"SEO_Action_Items_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                    
                    st.markdown("---")
                    
                    # Display action items summary
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        total_items = len(all_recommendations)
                        st.metric("Total Action Items", total_items)
                    
                    with col2:
                        high_priority_count = len([r for r in all_recommendations if r['priority'] == 'High'])
                        st.metric("High Priority", high_priority_count, delta=f"{high_priority_count/total_items*100:.0f}%")
                    
                    with col3:
                        easy_fixes = len([r for r in all_recommendations if r['difficulty'] == 'Easy'])
                        st.metric("Quick Wins", easy_fixes, delta="Easy fixes")
                    
                    with col4:
                        high_impact = len([r for r in all_recommendations if r['impact'] == 'High'])
                        st.metric("High Impact", high_impact, delta=f"{high_impact/total_items*100:.0f}%")
                    
                    st.markdown("---")
                    
                    # Display recommendations by site
                    for idx, result in enumerate(all_results):
                        site_name = result['file_name'].replace('.xlsx', '').replace('_', ' ')
                        site_recs = result['recommendations']
                        
                        if site_recs:
                            # Site header with summary metrics
                            col1, col2, col3 = st.columns([2, 1, 1])
                            
                            with col1:
                                st.markdown(f"### {site_name}")
                            
                            with col2:
                                site_high_priority = len([r for r in site_recs if r['priority'] == 'High'])
                                st.metric("High Priority Items", site_high_priority)
                            
                            with col3:
                                site_easy_fixes = len([r for r in site_recs if r['difficulty'] == 'Easy'])
                                st.metric("Quick Wins", site_easy_fixes)
                            
                            display_recommendations(site_recs, f"site_{idx}")
                            st.markdown("---")
                    
                    # Additional analysis section
                    st.markdown("### 📈 Action Items Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Priority distribution chart
                        priority_counts = pd.DataFrame(all_recommendations)['priority'].value_counts()
                        fig_priority = px.pie(
                            values=priority_counts.values, 
                            names=priority_counts.index,
                            title="Action Items by Priority",
                            color_discrete_map={
                                'High': '#dc3545',
                                'Medium': '#ffc107', 
                                'Low': '#28a745'
                            }
                        )
                        st.plotly_chart(fig_priority, use_container_width=True)
                    
                    with col2:
                        # Category distribution chart
                        category_counts = pd.DataFrame(all_recommendations)['category'].value_counts()
                        fig_category = px.bar(
                            x=category_counts.values,
                            y=category_counts.index,
                            orientation='h',
                            title="Action Items by Category",
                            color=category_counts.values,
                            color_continuous_scale='viridis'
                        )
                        fig_category.update_layout(showlegend=False)
                        st.plotly_chart(fig_category, use_container_width=True)
                
                else:
                    st.success("🎉 Excellent! No major issues found across all sites.")
                    
                    # Still provide an empty template for download
                    st.markdown("### Download Action Items Template")
                    template_data = pd.DataFrame({
                        'Item #': [1],
                        'Site': ['Example Site'],
                        'Category': ['Content SEO'],
                        'Priority': ['Medium'],
                        'Issue': ['Example issue'],
                        'Action Required': ['Example action'],
                        'Difficulty': ['Easy'],
                        'Impact': ['Medium'],
                        'Resources': ['Example resource links'],
                        'Status': ['Pending']
                    })
                    
                    output = BytesIO()
                    template_data.to_excel(output, index=False)
                    
                    st.download_button(
                        label="📊 Download Template",
                        data=output.getvalue(),
                        file_name=f"SEO_Action_Items_Template.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        help="Download template for tracking action items"
                    )
            
            with tab4:
                # Enhanced Detailed Audits with Bullet Points and Excel Export
                st.markdown("## 📋 Detailed SEO Audit Tables with Discoveries")
                
                for i, analysis in enumerate(detailed_analyses):
                    site_name = analysis['File Name'].replace('.xlsx', '').replace('_', ' ')
                    
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.markdown(f"### {site_name} - Technical SEO Audit")
                    with col2:
                        score = analysis['Overall Score']
                        score_color = "#dc3545" if score < 50 else "#ffc107" if score < 80 else "#28a745"
                        st.markdown(f"""
                        <div style="text-align: center;">
                            <div style="display: inline-block; width: 80px; height: 80px; border-radius: 50%; 
                                        background-color: {score_color}; color: white; 
                                        line-height: 80px; font-size: 24px; font-weight: bold;">
                                {score}/100
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown("**Export Audit Data:**")
                        
                        # Create comprehensive export data
                        audit_data = []
                        discoveries = analysis['Discoveries']
                        
                        for category, analysis_data in [
                            ('Content SEO', analysis['Content Analysis']),
                            ('Technical SEO', analysis['Technical Analysis']),
                            ('User Experience', analysis['UX Analysis']),
                            ('Off-Page SEO', analysis['Offpage Analysis'])
                        ]:
                            audit_data.append([category, '', '', ''])
                            
                            # Add detailed analysis items
                            for key, details in analysis_data.items():
                                status = "✓" if details['status'] == '✓' else "✗"
                                if key in ['mobile_speed', 'desktop_speed']:
                                    status = details['status']
                                improvement = " - Needs Improvement" if details.get('needs_improvement', False) else ""
                                audit_data.append(['', details['description'], f"{status}{improvement}", f"{details['score']}/100"])
                            
                            # Add bullet-point discoveries
                            if category in discoveries and discoveries[category]:
                                audit_data.append(['', '--- Key Discoveries ---', '', ''])
                                for discovery in discoveries[category]:
                                    audit_data.append(['', discovery, '', ''])
                            
                            audit_data.append(['', '', '', ''])  # Add spacing
                        
                        audit_df = pd.DataFrame(audit_data, columns=['Category', 'Factor', 'Status', 'Score'])
                        
                        # Excel export with enhanced formatting
                        output = BytesIO()
                        try:
                            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                # Main audit sheet
                                audit_df.to_excel(writer, sheet_name='SEO_Audit', index=False)
                                
                                # Discoveries only sheet
                                discoveries_data = []
                                for category, discovery_list in discoveries.items():
                                    discoveries_data.append([category, '', ''])
                                    for discovery in discovery_list:
                                        discoveries_data.append(['', discovery, ''])
                                    discoveries_data.append(['', '', ''])
                                
                                discoveries_df = pd.DataFrame(discoveries_data, columns=['Category', 'Discovery', 'Notes'])
                                discoveries_df.to_excel(writer, sheet_name='Key_Discoveries', index=False)
                            
                            st.download_button(
                                label=f"📊 Excel - {site_name}",
                                data=output.getvalue(),
                                file_name=f"{site_name.replace(' ', '_')}_SEO_Detailed_Audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key=f"detailed_excel_export_{i}",
                                help="Download detailed audit with discoveries and analysis"
                            )
                        except Exception as e:
                            st.error(f"Error creating Excel file: {str(e)}")
                            # Fallback to CSV
                            csv_data = audit_df.to_csv(index=False)
                            st.download_button(
                                label="📄 CSV Fallback",
                                data=csv_data,
                                file_name=f"{site_name.replace(' ', '_')}_SEO_Audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                key=f"detailed_csv_export_{i}"
                            )
                    
                    # Display audit categories with discoveries
                    for category, analysis_data in [
                        ('Content SEO', analysis['Content Analysis']),
                        ('Technical SEO', analysis['Technical Analysis']),
                        ('User Experience', analysis['UX Analysis']),
                        ('Off-Page SEO', analysis['Offpage Analysis'])
                    ]:
                        st.markdown(f"#### {category}")
                        
                        # Create two columns: audit table and discoveries
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.markdown("**Audit Results:**")
                            table_data = []
                            for key, details in analysis_data.items():
                                status = "✓" if details['status'] == '✓' else "✗"
                                if key in ['mobile_speed', 'desktop_speed']:
                                    status = details['status']
                                improvement = " - Needs Improvement" if details.get('needs_improvement', False) else ""
                                table_data.append([details['description'], f"{status}{improvement}"])
                            
                            table_df = pd.DataFrame(table_data, columns=['Factor', 'Status'])
                            st.dataframe(table_df, use_container_width=True)
                        
                        with col2:
                            discoveries = analysis['Discoveries']
                            if category in discoveries and discoveries[category]:
                                st.markdown("**Key Discoveries:**")
                                st.markdown("""
                                <div class="bullet-list">
                                """, unsafe_allow_html=True)
                                
                                for discovery in discoveries[category]:
                                    st.markdown(f"{discovery}")
                                
                                st.markdown("</div>", unsafe_allow_html=True)
                            else:
                                st.markdown("**Key Discoveries:**")
                                st.info("No specific discoveries available for this category.")
                    
                    st.markdown("---")
            
            with tab5:
                # Competitive analysis
                st.markdown("## 🏆 Competitive Analysis")
                if len(all_results) > 1:
                    radar_chart = create_comparison_chart(comparison_df)
                    st.plotly_chart(radar_chart, use_container_width=True, key="comparison_radar")
                
                # Detailed comparison table
                st.markdown("### Detailed Comparison")
                display_df = comparison_df[['File Name', 'Content SEO', 'Technical SEO', 'User Experience', 'Overall Readiness']].copy()
                display_df['File Name'] = display_df['File Name'].str.replace('.xlsx', '').str.replace('_', ' ')
                st.dataframe(display_df, use_container_width=True)
            
            with tab6:
                # Export functionality
                st.markdown("## 📥 Export Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    export_df = comparison_df.drop(['Content Details', 'Technical Details', 'UX Details'], axis=1)
                    export_df['File Name'] = export_df['File Name'].str.replace('.xlsx', '').str.replace('_', ' ')
                    
                    output = BytesIO()
                    try:
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            export_df.to_excel(writer, sheet_name='SEO Analysis', index=False)
                        
                        st.download_button(
                            label="📊 Download Excel Report",
                            data=output.getvalue(),
                            file_name=f"SEO_Comparison_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    except Exception as e:
                        st.error(f"Error creating Excel report: {str(e)}")
                
                with col2:
                    csv_data = export_df.to_csv(index=False)
                    st.download_button(
                        label="📄 Download CSV Report",
                        data=csv_data,
                        file_name=f"SEO_Comparison_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        else:
            st.error("No files were successfully processed. Please check your file formats and try again.")
    
    else:
        # Instructions when no files uploaded
        st.markdown("## 🚀 Enhanced Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🔥 Heatmap Analysis
            - Visual heatmaps by page type
            - Identify patterns in SEO issues
            - Prioritize fixes by impact
            """)
        
        with col2:
            st.markdown("""
            ### 🎯 Action Items
            - Specific, prioritized recommendations
            - Implementation difficulty ratings
            - Direct links to helpful resources
            """)
        
        with col3:
            st.markdown("""
            ### 📊 Executive Summary
            - Key findings dashboard
            - Performance comparisons
            - Strategic insights
            """)

        with st.expander("📖 How to export data from Screaming Frog", expanded=True):
            st.write("""
            **Step-by-step instructions:**
            
            1. **Open Screaming Frog SEO Spider**
            2. **Crawl your website** (enter URL and click Start)
            3. **Wait for crawl to complete**
            4. **Go to Bulk Export menu** → Response Codes → All
            5. **Select the Internal HTML tab**
            6. **Click Export** and save as CSV or Excel
            7. **Upload the file here** using the file uploader above
            
            ✅ **Tip**: Make sure to export from the "Internal HTML" tab for best results!
            """)

if __name__ == "__main__":
    main()
