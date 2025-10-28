-- Fashion Recommender Database Schema
-- PostgreSQL initialization script

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Skin tone analysis results
CREATE TABLE skin_tone_analysis (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    photo BYTEA NOT NULL,
    photo_filename VARCHAR(255),
    fitzpatrick_type VARCHAR(10) NOT NULL,
    undertone VARCHAR(20) NOT NULL,
    dominant_color_r INTEGER NOT NULL,
    dominant_color_g INTEGER NOT NULL,
    dominant_color_b INTEGER NOT NULL,
    confidence DECIMAL(3,2),
    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_user FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Wardrobes (collections of clothing)
CREATE TABLE wardrobes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Clothing items with image storage
CREATE TABLE clothing_items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    wardrobe_id UUID REFERENCES wardrobes(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    image BYTEA NOT NULL,
    image_filename VARCHAR(255),
    image_mimetype VARCHAR(100),
    category VARCHAR(50) NOT NULL, -- Topwear, Bottomwear, Dress, Footwear, Accessories
    clothing_type VARCHAR(100), -- Shirt, Jeans, etc
    dominant_color_r INTEGER,
    dominant_color_g INTEGER,
    dominant_color_b INTEGER,
    secondary_colors JSONB,
    style VARCHAR(50), -- Casual, Formal, etc
    pattern VARCHAR(50), -- Solid, Striped, etc
    season VARCHAR(20), -- Spring, Summer, Fall, Winter
    occasion VARCHAR(50), -- Party, Work, Casual, etc
    ai_analyzed BOOLEAN DEFAULT FALSE,
    ai_confidence DECIMAL(3,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Outfit recommendations
CREATE TABLE outfit_recommendations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    skin_tone_analysis_id UUID REFERENCES skin_tone_analysis(id),
    top_item_id UUID REFERENCES clothing_items(id),
    bottom_item_id UUID REFERENCES clothing_items(id),
    shoes_item_id UUID REFERENCES clothing_items(id),
    accessories_item_id UUID REFERENCES clothing_items(id),
    occasion VARCHAR(50) NOT NULL,
    season VARCHAR(20),
    compatibility_score DECIMAL(3,2) NOT NULL,
    skin_tone_match_score DECIMAL(3,2),
    color_harmony_type VARCHAR(50),
    reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    favorited BOOLEAN DEFAULT FALSE
);

-- Saved outfits (user favorites)
CREATE TABLE saved_outfits (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    outfit_recommendation_id UUID REFERENCES outfit_recommendations(id),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_skin_tone_user ON skin_tone_analysis(user_id);
CREATE INDEX idx_wardrobes_user ON wardrobes(user_id);
CREATE INDEX idx_clothing_user ON clothing_items(user_id);
CREATE INDEX idx_clothing_wardrobe ON clothing_items(wardrobe_id);
CREATE INDEX idx_clothing_category ON clothing_items(category);
CREATE INDEX idx_recommendations_user ON outfit_recommendations(user_id);
CREATE INDEX idx_recommendations_occasion ON outfit_recommendations(occasion);
CREATE INDEX idx_saved_outfits_user ON saved_outfits(user_id);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for automatic timestamp updates
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_wardrobes_updated_at BEFORE UPDATE ON wardrobes
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_clothing_items_updated_at BEFORE UPDATE ON clothing_items
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert default data
INSERT INTO users (email, name) VALUES 
    ('demo@example.com', 'Demo User');
