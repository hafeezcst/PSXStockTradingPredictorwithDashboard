#!/usr/bin/env python3
"""
Portfolio Report Verification Script
Verifies that portfolio reports are properly created and validates their contents
"""

import os
import sys
import json
import glob
from datetime import datetime, timedelta
from pathlib import Path

# Add current directory to path for imports
portfolio_mgmt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'src', 'data_processing', 'portfolio_management')
sys.path.append(portfolio_mgmt_path)
sys.path.append(os.path.join(portfolio_mgmt_path, 'core'))

class PortfolioReportVerifier:
    def __init__(self):
        self.project_root = self.get_project_root()
        self.exports_dir = os.path.join(self.project_root, 'data', 'exports')
        self.verification_results = {
            'total_checks': 0,
            'passed_checks': 0,
            'failed_checks': 0,
            'warnings': [],
            'errors': [],
            'files_found': [],
            'files_missing': []
        }
    
    def get_project_root(self):
        """Get project root directory"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Navigate up to project root
        project_root = os.path.join(current_dir, '..', '..', '..')
        return os.path.abspath(project_root)
    
    def check_exports_directory(self):
        """Check if exports directory exists"""
        print("🔍 Checking exports directory...")
        self.verification_results['total_checks'] += 1
        
        if os.path.exists(self.exports_dir):
            print(f"✅ Exports directory exists: {self.exports_dir}")
            self.verification_results['passed_checks'] += 1
            return True
        else:
            print(f"❌ Exports directory missing: {self.exports_dir}")
            self.verification_results['failed_checks'] += 1
            self.verification_results['errors'].append(f"Exports directory not found: {self.exports_dir}")
            return False
    
    def find_portfolio_reports(self, days_back=7):
        """Find portfolio reports from last N days"""
        print(f"\n🔍 Searching for portfolio reports from last {days_back} days...")
        
        json_pattern = os.path.join(self.exports_dir, "portfolio_report_*.json")
        txt_pattern = os.path.join(self.exports_dir, "portfolio_report_*.txt")
        
        json_files = glob.glob(json_pattern)
        txt_files = glob.glob(txt_pattern)
        
        # Filter by date (last N days)
        cutoff_date = datetime.now() - timedelta(days=days_back)
        recent_json = []
        recent_txt = []
        
        for file_path in json_files:
            try:
                # Extract timestamp from filename
                filename = os.path.basename(file_path)
                timestamp_str = filename.replace('portfolio_report_', '').replace('.json', '')
                file_date = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                
                if file_date >= cutoff_date:
                    recent_json.append({
                        'path': file_path,
                        'filename': filename,
                        'date': file_date,
                        'size': os.path.getsize(file_path)
                    })
            except:
                pass
        
        for file_path in txt_files:
            try:
                filename = os.path.basename(file_path)
                timestamp_str = filename.replace('portfolio_report_', '').replace('.txt', '')
                file_date = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                
                if file_date >= cutoff_date:
                    recent_txt.append({
                        'path': file_path,
                        'filename': filename,
                        'date': file_date,
                        'size': os.path.getsize(file_path)
                    })
            except:
                pass
        
        return recent_json, recent_txt
    
    def verify_specific_reports(self, target_timestamp="20250618_234453"):
        """Verify specific portfolio reports mentioned by user"""
        print(f"\n🎯 Verifying specific reports for timestamp: {target_timestamp}")
        
        json_file = os.path.join(self.exports_dir, f"portfolio_report_{target_timestamp}.json")
        txt_file = os.path.join(self.exports_dir, f"portfolio_report_{target_timestamp}.txt")
        
        results = {
            'json_exists': False,
            'txt_exists': False,
            'json_valid': False,
            'txt_valid': False,
            'json_size': 0,
            'txt_size': 0
        }
        
        # Check JSON file
        self.verification_results['total_checks'] += 1
        if os.path.exists(json_file):
            print(f"✅ JSON report found: {json_file}")
            results['json_exists'] = True
            results['json_size'] = os.path.getsize(json_file)
            self.verification_results['passed_checks'] += 1
            self.verification_results['files_found'].append(json_file)
            
            # Validate JSON content
            if self.validate_json_report(json_file):
                results['json_valid'] = True
        else:
            print(f"❌ JSON report missing: {json_file}")
            self.verification_results['failed_checks'] += 1
            self.verification_results['files_missing'].append(json_file)
        
        # Check TXT file
        self.verification_results['total_checks'] += 1
        if os.path.exists(txt_file):
            print(f"✅ TXT report found: {txt_file}")
            results['txt_exists'] = True
            results['txt_size'] = os.path.getsize(txt_file)
            self.verification_results['passed_checks'] += 1
            self.verification_results['files_found'].append(txt_file)
            
            # Validate TXT content
            if self.validate_txt_report(txt_file):
                results['txt_valid'] = True
        else:
            print(f"❌ TXT report missing: {txt_file}")
            self.verification_results['failed_checks'] += 1
            self.verification_results['files_missing'].append(txt_file)
        
        return results
    
    def validate_json_report(self, file_path):
        """Validate JSON report structure and content"""
        print(f"  🔍 Validating JSON content...")
        self.verification_results['total_checks'] += 1
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Check required fields
            required_fields = [
                'portfolio_value',
                'cash_balance',
                'invested_amount',
                'total_return_pct',
                'num_positions'
            ]
            
            missing_fields = []
            for field in required_fields:
                if field not in data:
                    missing_fields.append(field)
            
            if missing_fields:
                print(f"    ❌ Missing required fields: {missing_fields}")
                self.verification_results['failed_checks'] += 1
                self.verification_results['errors'].append(f"JSON missing fields: {missing_fields}")
                return False
            
            # Validate data types and ranges
            if not isinstance(data.get('portfolio_value'), (int, float)) or data['portfolio_value'] < 0:
                print(f"    ❌ Invalid portfolio_value: {data.get('portfolio_value')}")
                self.verification_results['failed_checks'] += 1
                return False
            
            if not isinstance(data.get('num_positions'), int) or data['num_positions'] < 0:
                print(f"    ❌ Invalid num_positions: {data.get('num_positions')}")
                self.verification_results['failed_checks'] += 1
                return False
            
            print(f"    ✅ JSON structure valid")
            print(f"    📊 Portfolio Value: {data.get('portfolio_value', 0):,.0f} PKR")
            print(f"    💰 Cash Balance: {data.get('cash_balance', 0):,.0f} PKR")
            print(f"    📈 Positions: {data.get('num_positions', 0)}")
            
            self.verification_results['passed_checks'] += 1
            return True
            
        except json.JSONDecodeError as e:
            print(f"    ❌ Invalid JSON format: {e}")
            self.verification_results['failed_checks'] += 1
            self.verification_results['errors'].append(f"JSON decode error: {e}")
            return False
        except Exception as e:
            print(f"    ❌ Error validating JSON: {e}")
            self.verification_results['failed_checks'] += 1
            self.verification_results['errors'].append(f"JSON validation error: {e}")
            return False
    
    def validate_txt_report(self, file_path):
        """Validate TXT report content"""
        print(f"  🔍 Validating TXT content...")
        self.verification_results['total_checks'] += 1
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check for required sections
            required_sections = [
                "PORTFOLIO MANAGEMENT SYSTEM",
                "Portfolio Value:",
                "Cash Balance:",
                "Invested Amount:",
                "Total Return:",
                "Number of Positions:"
            ]
            
            missing_sections = []
            for section in required_sections:
                if section not in content:
                    missing_sections.append(section)
            
            if missing_sections:
                print(f"    ❌ Missing required sections: {missing_sections}")
                self.verification_results['failed_checks'] += 1
                self.verification_results['errors'].append(f"TXT missing sections: {missing_sections}")
                return False
            
            # Check file size (should not be empty)
            file_size = os.path.getsize(file_path)
            if file_size < 100:  # Minimum expected size
                print(f"    ⚠️  TXT file seems too small: {file_size} bytes")
                self.verification_results['warnings'].append(f"TXT file suspiciously small: {file_size} bytes")
            
            print(f"    ✅ TXT structure valid")
            print(f"    📄 File size: {file_size} bytes")
            
            self.verification_results['passed_checks'] += 1
            return True
            
        except Exception as e:
            print(f"    ❌ Error validating TXT: {e}")
            self.verification_results['failed_checks'] += 1
            self.verification_results['errors'].append(f"TXT validation error: {e}")
            return False
    
    def test_report_generation(self):
        """Test if report generation is working"""
        print(f"\n🧪 Testing report generation...")
        
        try:
            # Import portfolio manager
            from core.simple_portfolio_manager import SimplePortfolioManager
            
            pm = SimplePortfolioManager()
            summary = pm.get_portfolio_summary()
            
            # Test directory creation
            test_exports_dir = os.path.join(self.project_root, 'data', 'exports', 'test')
            os.makedirs(test_exports_dir, exist_ok=True)
            
            # Generate test report
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            test_json_file = os.path.join(test_exports_dir, f"test_portfolio_report_{timestamp}.json")
            test_txt_file = os.path.join(test_exports_dir, f"test_portfolio_report_{timestamp}.txt")
            
            # Create JSON
            with open(test_json_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            # Create TXT
            with open(test_txt_file, 'w') as f:
                f.write("PORTFOLIO MANAGEMENT SYSTEM - TEST REPORT\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Report Generated: {datetime.now()}\n\n")
                f.write(f"Portfolio Value: {summary.get('portfolio_value', 0):,.0f} PKR\n")
                f.write(f"Cash Balance: {summary.get('cash_balance', 0):,.0f} PKR\n")
                f.write(f"Total Return: {summary.get('total_return_pct', 0):.2f}%\n")
                f.write(f"Number of Positions: {summary.get('num_positions', 0)}\n")
            
            # Verify test files
            if os.path.exists(test_json_file) and os.path.exists(test_txt_file):
                print(f"✅ Test report generation successful")
                print(f"  📄 Test JSON: {test_json_file}")
                print(f"  📄 Test TXT: {test_txt_file}")
                
                # Clean up test files
                os.remove(test_json_file)
                os.remove(test_txt_file)
                os.rmdir(test_exports_dir)
                
                self.verification_results['total_checks'] += 1
                self.verification_results['passed_checks'] += 1
                return True
            else:
                print(f"❌ Test report generation failed")
                self.verification_results['total_checks'] += 1
                self.verification_results['failed_checks'] += 1
                return False
                
        except Exception as e:
            print(f"❌ Error testing report generation: {e}")
            self.verification_results['total_checks'] += 1
            self.verification_results['failed_checks'] += 1
            self.verification_results['errors'].append(f"Test generation error: {e}")
            return False
    
    def check_permissions(self):
        """Check directory permissions for writing reports"""
        print(f"\n🔐 Checking directory permissions...")
        self.verification_results['total_checks'] += 1
        
        try:
            # Test write permission
            test_file = os.path.join(self.exports_dir, 'permission_test.txt')
            with open(test_file, 'w') as f:
                f.write("Permission test")
            
            # Clean up
            os.remove(test_file)
            
            print(f"✅ Write permissions OK for: {self.exports_dir}")
            self.verification_results['passed_checks'] += 1
            return True
            
        except Exception as e:
            print(f"❌ Permission error: {e}")
            self.verification_results['failed_checks'] += 1
            self.verification_results['errors'].append(f"Permission error: {e}")
            return False
    
    def generate_verification_report(self):
        """Generate comprehensive verification report"""
        print(f"\n" + "="*80)
        print("📋 PORTFOLIO REPORT VERIFICATION SUMMARY")
        print("="*80)
        
        # Overall statistics
        total = self.verification_results['total_checks']
        passed = self.verification_results['passed_checks']
        failed = self.verification_results['failed_checks']
        success_rate = (passed / total * 100) if total > 0 else 0
        
        print(f"\n📊 VERIFICATION STATISTICS:")
        print(f"Total Checks: {total}")
        print(f"Passed: {passed} ✅")
        print(f"Failed: {failed} ❌")
        print(f"Success Rate: {success_rate:.1f}%")
        
        # Files found
        if self.verification_results['files_found']:
            print(f"\n📁 FILES FOUND:")
            for file_path in self.verification_results['files_found']:
                size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                print(f"  ✅ {file_path} ({size} bytes)")
        
        # Files missing
        if self.verification_results['files_missing']:
            print(f"\n❌ FILES MISSING:")
            for file_path in self.verification_results['files_missing']:
                print(f"  ❌ {file_path}")
        
        # Warnings
        if self.verification_results['warnings']:
            print(f"\n⚠️  WARNINGS:")
            for warning in self.verification_results['warnings']:
                print(f"  ⚠️  {warning}")
        
        # Errors
        if self.verification_results['errors']:
            print(f"\n🚨 ERRORS:")
            for error in self.verification_results['errors']:
                print(f"  🚨 {error}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if failed > 0:
            print(f"  📝 Fix the {failed} failed checks above")
            print(f"  🔧 Run the portfolio export function manually to test")
            print(f"  📂 Ensure data/exports directory has write permissions")
            print(f"  🔍 Check if portfolio management system is properly configured")
        else:
            print(f"  ✅ All checks passed! Portfolio reporting system is working correctly.")
        
        return success_rate == 100.0

    def run_comprehensive_verification(self):
        """Run all verification checks"""
        print("🚀 STARTING PORTFOLIO REPORT VERIFICATION")
        print("="*80)
        
        # Check exports directory
        if not self.check_exports_directory():
            print("\n❌ Critical: Exports directory missing. Creating it...")
            os.makedirs(self.exports_dir, exist_ok=True)
        
        # Check permissions
        self.check_permissions()
        
        # Find recent reports
        recent_json, recent_txt = self.find_portfolio_reports()
        
        print(f"\n📊 RECENT REPORTS FOUND:")
        print(f"JSON Reports: {len(recent_json)}")
        print(f"TXT Reports: {len(recent_txt)}")
        
        if recent_json:
            print(f"\n📄 Recent JSON Reports:")
            for report in sorted(recent_json, key=lambda x: x['date'], reverse=True)[:5]:
                print(f"  📅 {report['filename']} - {report['date']} ({report['size']} bytes)")
        
        if recent_txt:
            print(f"\n📄 Recent TXT Reports:")
            for report in sorted(recent_txt, key=lambda x: x['date'], reverse=True)[:5]:
                print(f"  📅 {report['filename']} - {report['date']} ({report['size']} bytes)")
        
        # Verify specific reports mentioned by user
        specific_results = self.verify_specific_reports()
        
        # Test report generation capability
        self.test_report_generation()
        
        # Generate final report
        all_passed = self.generate_verification_report()
        
        return all_passed

def main():
    """Main execution function"""
    try:
        verifier = PortfolioReportVerifier()
        success = verifier.run_comprehensive_verification()
        
        print(f"\n🎯 VERIFICATION COMPLETE")
        if success:
            print("✅ All checks passed! Portfolio reporting system is working correctly.")
            return 0
        else:
            print("❌ Some checks failed. Review the issues above.")
            return 1
            
    except Exception as e:
        print(f"\n🚨 VERIFICATION ERROR: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
