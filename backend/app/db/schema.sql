-- Database schema for PSX Investment Application

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- KMI-30 Stocks table
CREATE TABLE kmi30_stock (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(10) NOT NULL UNIQUE,
    name VARCHAR(100) NOT NULL,
    sector VARCHAR(50),
    market_cap DECIMAL(20,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Users table
CREATE TABLE "user" (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) NOT NULL UNIQUE,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(100),
    is_active BOOLEAN DEFAULT true,
    is_superuser BOOLEAN DEFAULT false,
    subscription_tier VARCHAR(20) DEFAULT 'free',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Portfolios table
CREATE TABLE portfolio (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES "user"(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Portfolio Holdings table
CREATE TABLE portfolio_holding (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    portfolio_id UUID REFERENCES portfolio(id) ON DELETE CASCADE,
    stock_id UUID REFERENCES kmi30_stock(id) ON DELETE CASCADE,
    shares INTEGER NOT NULL,
    purchase_price DECIMAL(10,2) NOT NULL,
    entry_date DATE NOT NULL,
    take_profit DECIMAL(10,2),
    stop_loss DECIMAL(10,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Signal tables (partitioned by year)
CREATE TABLE buy_signal (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    stock_id UUID REFERENCES kmi30_stock(id) ON DELETE CASCADE,
    signal_date DATE NOT NULL,
    technical_score DECIMAL(5,2),
    fundamental_score DECIMAL(5,2),
    tradingview_signal VARCHAR(20),
    volume BIGINT,
    rsi_weekly DECIMAL(5,2),
    rsi_monthly DECIMAL(5,2),
    rsi_quarterly DECIMAL(5,2),
    ao_cross_positive BOOLEAN,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
) PARTITION BY RANGE (signal_date);

CREATE TABLE sell_signal (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    stock_id UUID REFERENCES kmi30_stock(id) ON DELETE CASCADE,
    signal_date DATE NOT NULL,
    technical_score DECIMAL(5,2),
    fundamental_score DECIMAL(5,2),
    tradingview_signal VARCHAR(20),
    stop_loss_triggered BOOLEAN DEFAULT false,
    take_profit_triggered BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
) PARTITION BY RANGE (signal_date);

CREATE TABLE neutral_signal (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    stock_id UUID REFERENCES kmi30_stock(id) ON DELETE CASCADE,
    signal_date DATE NOT NULL,
    technical_score DECIMAL(5,2),
    fundamental_score DECIMAL(5,2),
    tradingview_signal VARCHAR(20),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
) PARTITION BY RANGE (signal_date);

-- Signal transitions
CREATE TABLE signal_transition (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    stock_id UUID REFERENCES kmi30_stock(id) ON DELETE CASCADE,
    from_signal_type VARCHAR(20) NOT NULL,
    to_signal_type VARCHAR(20) NOT NULL,
    transition_date DATE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Dividends table
CREATE TABLE dividend (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    stock_id UUID REFERENCES kmi30_stock(id) ON DELETE CASCADE,
    declaration_date DATE NOT NULL,
    payment_date DATE,
    amount DECIMAL(10,4) NOT NULL,
    yield DECIMAL(5,2),
    source VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Fund investments table
CREATE TABLE fund_investment (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    stock_id UUID REFERENCES kmi30_stock(id) ON DELETE CASCADE,
    fund_name VARCHAR(100) NOT NULL,
    ownership_percentage DECIMAL(5,2),
    source VARCHAR(50),
    last_updated DATE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ETF membership table
CREATE TABLE etf_membership (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    stock_id UUID REFERENCES kmi30_stock(id) ON DELETE CASCADE,
    etf_name VARCHAR(100) NOT NULL,
    weight DECIMAL(5,2),
    source VARCHAR(50),
    last_updated DATE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- User activity log
CREATE TABLE user_activity_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES "user"(id) ON DELETE CASCADE,
    activity_type VARCHAR(50) NOT NULL,
    description TEXT,
    ip_address VARCHAR(45),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- API usage tracking
CREATE TABLE api_usage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES "user"(id) ON DELETE CASCADE,
    api_name VARCHAR(50) NOT NULL,
    request_count INTEGER DEFAULT 0,
    last_used TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Analytics reports
CREATE TABLE analytics_report (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES "user"(id) ON DELETE CASCADE,
    report_type VARCHAR(50) NOT NULL,
    report_data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Content queue for social media
CREATE TABLE content_queue (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content_type VARCHAR(50) NOT NULL,
    content_data JSONB NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    platform VARCHAR(50),
    scheduled_time TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Blockchain transactions
CREATE TABLE blockchain_transaction (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES "user"(id) ON DELETE CASCADE,
    transaction_hash VARCHAR(66) NOT NULL,
    transaction_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_stock_symbol ON kmi30_stock(symbol);
CREATE INDEX idx_user_email ON "user"(email);
CREATE INDEX idx_portfolio_user ON portfolio(user_id);
CREATE INDEX idx_holding_portfolio ON portfolio_holding(portfolio_id);
CREATE INDEX idx_holding_stock ON portfolio_holding(stock_id);
CREATE INDEX idx_signal_stock ON buy_signal(stock_id);
CREATE INDEX idx_signal_date ON buy_signal(signal_date);
CREATE INDEX idx_dividend_stock ON dividend(stock_id);
CREATE INDEX idx_fund_stock ON fund_investment(stock_id);
CREATE INDEX idx_etf_stock ON etf_membership(stock_id);
CREATE INDEX idx_activity_user ON user_activity_log(user_id);
CREATE INDEX idx_api_user ON api_usage(user_id);
CREATE INDEX idx_report_user ON analytics_report(user_id);
CREATE INDEX idx_content_status ON content_queue(status);
CREATE INDEX idx_blockchain_user ON blockchain_transaction(user_id);

-- Create partitions for signal tables (example for 2025)
CREATE TABLE buy_signal_2025 PARTITION OF buy_signal
    FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');

CREATE TABLE sell_signal_2025 PARTITION OF sell_signal
    FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');

CREATE TABLE neutral_signal_2025 PARTITION OF neutral_signal
    FOR VALUES FROM ('2025-01-01') TO ('2026-01-01'); 