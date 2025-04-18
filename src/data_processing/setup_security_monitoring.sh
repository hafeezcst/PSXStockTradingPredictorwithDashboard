#!/bin/bash

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WRAPPER_SCRIPT="script_security_wrapper.py"
BACKUP_DIR="data/backups/scripts"
LOG_DIR="logs/security"
SERVICE_NAME="psx_script_security"

# Create required directories
mkdir -p "$BACKUP_DIR"
mkdir -p "$LOG_DIR"

# Determine OS and setup appropriate service
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux systemd service setup
    SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
    
    echo "[Unit]
Description=PSX Script Security Monitor
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$SCRIPT_DIR
ExecStart=/usr/bin/python3 ${SCRIPT_DIR}/${WRAPPER_SCRIPT}
Restart=always
RestartSec=60

[Install]
WantedBy=multi-user.target" | sudo tee "$SERVICE_FILE" > /dev/null

    sudo systemctl daemon-reload
    sudo systemctl enable "$SERVICE_NAME"
    sudo systemctl start "$SERVICE_NAME"
    
    echo "Linux systemd service installed and started"
    
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows scheduled task setup
    TASK_NAME="PSX Script Security Monitor"
    SCHTASKS_CMD="schtasks /Create /TN \"$TASK_NAME\" /TR \"python.exe ${SCRIPT_DIR//\//\\}\\${WRAPPER_SCRIPT}\" /SC HOURLY /ST 00:00 /RU $(whoami) /F"
    
    eval "$SCHTASKS_CMD"
    echo "Windows scheduled task created to run hourly"
    
else
    echo "Unsupported OS: $OSTYPE"
    exit 1
fi

# Set up log rotation
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    LOGROTATE_FILE="/etc/logrotate.d/${SERVICE_NAME}"
    
    echo "${SCRIPT_DIR}/script_security.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
}" | sudo tee "$LOGROTATE_FILE" > /dev/null
    
    echo "Log rotation configured for Linux"
fi

echo "Security monitoring setup complete"