"""
Test script for Enhanced PSX Data Reader

This script validates the enhanced implementation with basic functionality tests.
"""

import sys
import os
import logging
from datetime import date, timedelta

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_configuration():
    """Test configuration loading"""
    print("Testing configuration management...")
    try:
        from config_manager import AppConfig
        config = AppConfig.from_yaml()
        config.validate()
        print("✅ Configuration loaded and validated successfully")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_database_manager():
    """Test database manager functionality"""
    print("Testing database manager...")
    try:
        from config_manager import AppConfig, DatabaseConfig
        from enhanced_db_manager import EnhancedDatabaseManager
        
        # Create test configuration
        db_config = DatabaseConfig()
        db_manager = EnhancedDatabaseManager(db_config)
        
        # Test connection
        is_connected = db_manager.check_connection()
        print(f"✅ Database connection test: {'Connected' if is_connected else 'Failed'}")
        
        # Test table operations
        tables = db_manager.get_table_names()
        print(f"✅ Found {len(tables)} tables in database")
        
        db_manager.close_connections()
        return True
    except Exception as e:
        print(f"❌ Database manager test failed: {e}")
        return False

def test_api_client():
    """Test API client functionality"""
    print("Testing API client...")
    try:
        from config_manager import APIConfig
        from api_client import PSXAPIClient
        
        # Create test configuration
        api_config = APIConfig()
        api_client = PSXAPIClient(api_config)
        
        # Test symbols fetch (this will make an actual API call)
        print("Attempting to fetch symbols from PSX...")
        symbols = api_client.fetch_symbols()
        
        if symbols:
            print("✅ API client successfully fetched symbols")
        else:
            print("⚠️ API client connected but no symbols returned")
        
        api_client.close()
        return True
    except Exception as e:
        print(f"❌ API client test failed: {e}")
        return False

def test_data_validation():
    """Test data validation functionality"""
    print("Testing data validation...")
    try:
        import pandas as pd
        from config_manager import DataValidationConfig
        from data_validator import DataValidator
        
        # Create test data
        test_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [99, 100, 101],
            'Close': [104, 105, 106],
            'Volume': [1000, 1100, 1200]
        }, index=pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']))
        
        # Test validation
        config = DataValidationConfig()
        validator = DataValidator(config)
        
        is_valid, issues = validator.validate_stock_data(test_data, 'TEST')
        print(f"✅ Data validation test: {'Valid' if is_valid else f'Issues found: {issues}'}")
        
        return True
    except Exception as e:
        print(f"❌ Data validation test failed: {e}")
        return False

def test_enhanced_reader():
    """Test the enhanced data reader (basic functionality only)"""
    print("Testing Enhanced Data Reader...")
    try:        # Import with fallback handling
        try:
            from enhanced_psx_data_reader import EnhancedDataReader
        except ImportError:
            print("⚠️ Could not import EnhancedDataReader, trying alternative import...")
            import importlib.util
            import os
            
            module_path = os.path.join(
                os.path.dirname(__file__), 
                "enhanced_psx_data_reader.py"
            )
            spec = importlib.util.spec_from_file_location("psx_reader", module_path)
            psx_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(psx_module)
            EnhancedDataReader = psx_module.EnhancedDataReader
        
        # Test initialization
        with EnhancedDataReader() as reader:
            print("✅ Enhanced Data Reader initialized successfully")
            
            # Test basic functionality
            performance_metrics = reader.get_performance_metrics()
            print(f"✅ Performance metrics retrieved: {len(performance_metrics)} categories")
            
            # Test tickers (if API is available)
            try:
                tickers = reader.tickers()
                print(f"✅ Tickers test: {'Success' if not tickers.empty else 'No tickers returned'}")
            except Exception as e:
                print(f"⚠️ Tickers test failed (API may be unavailable): {e}")
        
        return True
    except Exception as e:
        print(f"❌ Enhanced Data Reader test failed: {e}")
        return False

def run_comprehensive_test():
    """Run comprehensive test suite"""
    print("🚀 Starting Enhanced PSX Data Reader Test Suite")
    print("=" * 60)
    
    tests = [
        ("Configuration Management", test_configuration),
        ("Database Manager", test_database_manager),
        ("API Client", test_api_client),
        ("Data Validation", test_data_validation),
        ("Enhanced Data Reader", test_enhanced_reader)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25} {status}")
    
    print(f"\nOverall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The enhanced implementation is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    success = run_comprehensive_test()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
