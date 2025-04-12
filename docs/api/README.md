# PSX Stock Trading Predictor API Documentation

## Overview
This document provides detailed information about the PSX Stock Trading Predictor API endpoints, request/response formats, and authentication requirements.

## Base URL
```
http://localhost:5000/api/v1
```

## Authentication
Currently, the API does not require authentication for development purposes. In production, API key authentication will be implemented.

## Endpoints

### Stock Data

#### Get Stock Data
```http
GET /stocks/{symbol}
```

Query Parameters:
- `start_date` (optional): Start date for historical data (YYYY-MM-DD)
- `end_date` (optional): End date for historical data (YYYY-MM-DD)

Response:
```json
{
    "symbol": "string",
    "data": [
        {
            "date": "YYYY-MM-DD",
            "open": "float",
            "high": "float",
            "low": "float",
            "close": "float",
            "volume": "integer"
        }
    ]
}
```

### Predictions

#### Get Stock Prediction
```http
GET /predictions/{symbol}
```

Query Parameters:
- `days` (optional): Number of days to predict (default: 30)

Response:
```json
{
    "symbol": "string",
    "predictions": [
        {
            "date": "YYYY-MM-DD",
            "predicted_price": "float",
            "confidence": "float"
        }
    ]
}
```

## Error Responses

### 400 Bad Request
```json
{
    "error": "Invalid request parameters",
    "details": "string"
}
```

### 404 Not Found
```json
{
    "error": "Resource not found",
    "details": "string"
}
```

### 500 Internal Server Error
```json
{
    "error": "Internal server error",
    "details": "string"
}
``` 