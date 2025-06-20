#!/usr/bin/env python3
"""
Quick Portfolio Report Verification
Simple script to check if portfolio reports are being created properly
"""

import os
import json
import glob
from datetime import datetime, timedelta

def verify_portfolio_reports(target_timestamp=None, days_back=1):
    """
    Verify portfolio reports exist and are valid
    
    Args:
        target_timestamp: Specific timestamp to check (e.g., "20250618_234453")
        days_back: How many days back to search for reports
    
    Returns:
        dict: Verification results
    """
    
    # Get project root and exports directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, '..', '..', '..', '..')
    project_root = os.path.abspath(project_root)
    exports_dir = os.path.join(project_root, 'data', 'exports')
    
    results = {
        'exports_dir_exists': False,
        'target_json_exists': False,
        'target_txt_exists': False,
        'json_valid': False,
        'txt_valid': False,
        'recent_reports_count': 0,
        'errors': [],
        'warnings': [],
        'files_found': []
    }
    
    print(f"🔍 Verifying Portfolio Reports...")
    print(f"📂 Exports directory: {exports_dir}")
    
    # Check if exports directory exists
    if os.path.exists(exports_dir):
        results['exports_dir_exists'] = True
        print(f"✅ Exports directory exists")
    else:
        results['errors'].append("Exports directory does not exist")
        print(f"❌ Exports directory missing")
        return results
    
    # Check specific timestamp if provided
    if target_timestamp:
        print(f"\n🎯 Checking specific reports for timestamp: {target_timestamp}")
        
        json_file = os.path.join(exports_dir, f"portfolio_report_{target_timestamp}.json")
        txt_file = os.path.join(exports_dir, f"portfolio_report_{target_timestamp}.txt")
        
        # Check JSON file
        if os.path.exists(json_file):
            results['target_json_exists'] = True
            results['files_found'].append(json_file)
            print(f"✅ JSON report found: {os.path.basename(json_file)}")
            
            # Validate JSON
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Basic validation
                required_fields = ['portfolio_value', 'cash_balance', 'num_positions']
                if all(field in data for field in required_fields):
                    results['json_valid'] = True
                    print(f"  ✅ JSON structure valid")
                    print(f"  📊 Portfolio Value: {data.get('portfolio_value', 0):,.0f} PKR")
                    print(f"  💰 Cash Balance: {data.get('cash_balance', 0):,.0f} PKR")
                else:
                    results['errors'].append("JSON missing required fields")
                    print(f"  ❌ JSON missing required fields")
                    
            except Exception as e:
                results['errors'].append(f"JSON validation error: {e}")
                print(f"  ❌ JSON validation failed: {e}")
        else:
            results['errors'].append(f"JSON report not found: {json_file}")
            print(f"❌ JSON report missing: portfolio_report_{target_timestamp}.json")
        
        # Check TXT file
        if os.path.exists(txt_file):
            results['target_txt_exists'] = True
            results['files_found'].append(txt_file)
            print(f"✅ TXT report found: {os.path.basename(txt_file)}")
            
            # Validate TXT
            try:
                with open(txt_file, 'r') as f:
                    content = f.read()
                
                if len(content) > 100 and "PORTFOLIO MANAGEMENT SYSTEM" in content:
                    results['txt_valid'] = True
                    print(f"  ✅ TXT structure valid ({len(content)} characters)")
                else:
                    results['errors'].append("TXT file invalid or too short")
                    print(f"  ❌ TXT file seems invalid")
                    
            except Exception as e:
                results['errors'].append(f"TXT validation error: {e}")
                print(f"  ❌ TXT validation failed: {e}")
        else:
            results['errors'].append(f"TXT report not found: {txt_file}")
            print(f"❌ TXT report missing: portfolio_report_{target_timestamp}.txt")
    
    # Check for recent reports
    print(f"\n📅 Checking for recent reports (last {days_back} days)...")
    
    cutoff_date = datetime.now() - timedelta(days=days_back)
    json_pattern = os.path.join(exports_dir, "portfolio_report_*.json")
    txt_pattern = os.path.join(exports_dir, "portfolio_report_*.txt")
    
    recent_files = []
    
    # Find JSON files
    for file_path in glob.glob(json_pattern):
        try:
            # Extract timestamp from filename
            filename = os.path.basename(file_path)
            timestamp_str = filename.replace('portfolio_report_', '').replace('.json', '')
            file_date = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
            
            if file_date >= cutoff_date:
                file_size = os.path.getsize(file_path)
                recent_files.append({
                    'path': file_path,
                    'filename': filename,
                    'date': file_date,
                    'size': file_size,
                    'type': 'JSON'
                })
        except:
            pass
    
    # Find TXT files  
    for file_path in glob.glob(txt_pattern):
        try:
            filename = os.path.basename(file_path)
            timestamp_str = filename.replace('portfolio_report_', '').replace('.txt', '')
            file_date = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
            
            if file_date >= cutoff_date:
                file_size = os.path.getsize(file_path)
                recent_files.append({
                    'path': file_path,
                    'filename': filename,
                    'date': file_date,
                    'size': file_size,
                    'type': 'TXT'
                })
        except:
            pass
    
    results['recent_reports_count'] = len(recent_files)
    
    if recent_files:
        print(f"✅ Found {len(recent_files)} recent report files:")
        for file_info in sorted(recent_files, key=lambda x: x['date'], reverse=True):
            print(f"  📄 {file_info['filename']} ({file_info['type']}, {file_info['size']} bytes)")
            results['files_found'].append(file_info['path'])
    else:
        print(f"⚠️  No recent reports found in last {days_back} days")
        results['warnings'].append(f"No recent reports found in last {days_back} days")
    
    # Summary
    print(f"\n📋 VERIFICATION SUMMARY:")
    print(f"Exports Directory: {'✅' if results['exports_dir_exists'] else '❌'}")
    if target_timestamp:
        print(f"Target JSON ({target_timestamp}): {'✅' if results['target_json_exists'] else '❌'}")
        print(f"Target TXT ({target_timestamp}): {'✅' if results['target_txt_exists'] else '❌'}")
    print(f"Recent Reports: {results['recent_reports_count']} found")
    print(f"Errors: {len(results['errors'])}")
    print(f"Warnings: {len(results['warnings'])}")
    
    if results['errors']:
        print(f"\n🚨 ERRORS:")
        for error in results['errors']:
            print(f"  • {error}")
    
    if results['warnings']:
        print(f"\n⚠️  WARNINGS:")
        for warning in results['warnings']:
            print(f"  • {warning}")
    
    return results

def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify Portfolio Reports')
    parser.add_argument('--timestamp', help='Specific timestamp to check (e.g., 20250618_234453)')
    parser.add_argument('--days', type=int, default=7, help='Days back to search for reports')
    
    args = parser.parse_args()
    
    print("🚀 PORTFOLIO REPORT VERIFICATION TOOL")
    print("="*50)
    
    results = verify_portfolio_reports(
        target_timestamp=args.timestamp,
        days_back=args.days
    )
    
    # Return appropriate exit code
    if results['errors']:
        print(f"\n❌ Verification completed with errors")
        return 1
    elif results['warnings']:
        print(f"\n⚠️  Verification completed with warnings")
        return 0
    else:
        print(f"\n✅ Verification completed successfully")
        return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
