# Code Protection Guidelines

## Protected Files
- `draw_indicator_trend_lines.py` (Version 1.0)
  - Status: Frozen for production use
  - Last modified: 2025-04-01
  - Protection measures:
    - Git tag v1.0-stable created
    - Read-only file permissions set (chmod 444)
    - Warning header added to file
    - Archived backup copy created: `draw_indicator_trend_lines_v1.0_20250401.py`

## Modification Process
1. Create a working copy with new version number
2. Make changes in the working copy only
3. Test changes thoroughly in development environment
4. Get approval from project lead
5. Update documentation with change details
6. Only then replace production version following these steps:
   - Remove read-only permissions
   - Make backup of current production version
   - Deploy new version
   - Set read-only permissions on new version
   - Create new git tag
## Enhanced Protection Features
- Checksum verification: Run `shasum -a 256 -c scripts/data_processing/draw_indicator_trend_lines.sha256` to verify file integrity
- Git protection: Pre-commit hook prevents accidental commits to protected file
- CI/CD integration: Add `scripts/verify_protected_files.sh` to your build pipeline
