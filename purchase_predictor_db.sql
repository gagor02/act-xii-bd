-- ================================================
-- ACT XII: Database Schema for Purchase Predictor
-- ================================================

-- Create database (run this separately if needed)
-- CREATE DATABASE purchase_predictor_db9b;

-- Connect to database
-- \c purchase_predictor_db9b;

-- Table to store prediction history
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    gender VARCHAR(10) NOT NULL,
    age INTEGER NOT NULL,
    estimated_salary NUMERIC(10,2) NOT NULL,
    predicted_purchase BOOLEAN NOT NULL,
    purchase_probability NUMERIC(5,4) NOT NULL,
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table to store training data
CREATE TABLE IF NOT EXISTS user_data (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    gender VARCHAR(10) NOT NULL,
    age INTEGER NOT NULL,
    estimated_salary NUMERIC(10,2) NOT NULL,
    purchased BOOLEAN NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_prediction_date ON predictions(prediction_date);
CREATE INDEX IF NOT EXISTS idx_predicted_purchase ON predictions(predicted_purchase);
CREATE INDEX IF NOT EXISTS idx_age_salary ON predictions(age, estimated_salary);

-- Insert sample training data
INSERT INTO user_data (user_id, gender, age, estimated_salary, purchased) VALUES
(15624510, 'Male', 19, 19000, FALSE),
(15810944, 'Male', 35, 20000, FALSE),
(15668575, 'Female', 26, 43000, FALSE),
(15603246, 'Female', 27, 57000, FALSE),
(15804002, 'Male', 19, 76000, FALSE),
(15728773, 'Male', 27, 58000, FALSE),
(15598044, 'Female', 27, 84000, FALSE),
(15694829, 'Female', 32, 150000, TRUE),
(15600575, 'Male', 25, 33000, FALSE),
(15727311, 'Female', 35, 65000, FALSE),
(15570769, 'Female', 26, 80000, FALSE),
(15606274, 'Female', 26, 52000, FALSE),
(15746139, 'Male', 20, 86000, FALSE),
(15704987, 'Male', 32, 18000, FALSE),
(15628972, 'Male', 18, 82000, FALSE);

-- View to get prediction statistics
CREATE OR REPLACE VIEW prediction_stats AS
SELECT 
    COUNT(*) as total_predictions,
    SUM(CASE WHEN predicted_purchase THEN 1 ELSE 0 END) as predicted_purchases,
    AVG(age) as avg_age,
    AVG(estimated_salary) as avg_salary,
    AVG(purchase_probability) as avg_probability
FROM predictions;

-- View for gender-based analytics
CREATE OR REPLACE VIEW gender_analytics AS
SELECT 
    gender,
    COUNT(*) as total_predictions,
    SUM(CASE WHEN predicted_purchase THEN 1 ELSE 0 END) as predicted_purchases,
    ROUND(AVG(purchase_probability) * 100, 2) as avg_probability_pct
FROM predictions
GROUP BY gender;

COMMENT ON TABLE predictions IS 'Stores all purchase predictions made by the model';
COMMENT ON TABLE user_data IS 'Stores the training dataset';
COMMENT ON VIEW prediction_stats IS 'Aggregated statistics of all predictions';
COMMENT ON VIEW gender_analytics IS 'Analytics broken down by gender';