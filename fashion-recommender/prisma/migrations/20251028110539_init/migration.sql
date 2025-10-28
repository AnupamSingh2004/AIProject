-- CreateTable
CREATE TABLE "users" (
    "id" UUID NOT NULL,
    "email" VARCHAR(255) NOT NULL,
    "name" VARCHAR(255),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "users_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "skin_tone_analysis" (
    "id" UUID NOT NULL,
    "user_id" UUID NOT NULL,
    "photo" BYTEA NOT NULL,
    "photo_filename" VARCHAR(255),
    "fitzpatrick_type" VARCHAR(10) NOT NULL,
    "undertone" VARCHAR(20) NOT NULL,
    "dominant_color_r" INTEGER NOT NULL,
    "dominant_color_g" INTEGER NOT NULL,
    "dominant_color_b" INTEGER NOT NULL,
    "confidence" DECIMAL(3,2),
    "analyzed_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "skin_tone_analysis_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "wardrobes" (
    "id" UUID NOT NULL,
    "user_id" UUID NOT NULL,
    "name" VARCHAR(255) NOT NULL,
    "description" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "wardrobes_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "clothing_items" (
    "id" UUID NOT NULL,
    "wardrobe_id" UUID NOT NULL,
    "user_id" UUID NOT NULL,
    "name" VARCHAR(255) NOT NULL,
    "image" BYTEA NOT NULL,
    "image_filename" VARCHAR(255),
    "image_mimetype" VARCHAR(100),
    "category" VARCHAR(50) NOT NULL,
    "clothing_type" VARCHAR(100),
    "dominant_color_r" INTEGER,
    "dominant_color_g" INTEGER,
    "dominant_color_b" INTEGER,
    "secondary_colors" JSONB,
    "style" VARCHAR(50),
    "pattern" VARCHAR(50),
    "season" VARCHAR(20),
    "occasion" VARCHAR(50),
    "ai_analyzed" BOOLEAN NOT NULL DEFAULT false,
    "ai_confidence" DECIMAL(3,2),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "clothing_items_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "outfit_recommendations" (
    "id" UUID NOT NULL,
    "user_id" UUID NOT NULL,
    "skin_tone_analysis_id" UUID,
    "top_item_id" UUID,
    "bottom_item_id" UUID,
    "shoes_item_id" UUID,
    "accessories_item_id" UUID,
    "occasion" VARCHAR(50) NOT NULL,
    "season" VARCHAR(20),
    "compatibility_score" DECIMAL(3,2) NOT NULL,
    "skin_tone_match_score" DECIMAL(3,2),
    "color_harmony_type" VARCHAR(50),
    "reason" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "favorited" BOOLEAN NOT NULL DEFAULT false,

    CONSTRAINT "outfit_recommendations_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "saved_outfits" (
    "id" UUID NOT NULL,
    "user_id" UUID NOT NULL,
    "name" VARCHAR(255) NOT NULL,
    "outfit_recommendation_id" UUID,
    "notes" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "saved_outfits_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "users_email_key" ON "users"("email");

-- CreateIndex
CREATE INDEX "skin_tone_analysis_user_id_idx" ON "skin_tone_analysis"("user_id");

-- CreateIndex
CREATE INDEX "wardrobes_user_id_idx" ON "wardrobes"("user_id");

-- CreateIndex
CREATE INDEX "clothing_items_user_id_idx" ON "clothing_items"("user_id");

-- CreateIndex
CREATE INDEX "clothing_items_wardrobe_id_idx" ON "clothing_items"("wardrobe_id");

-- CreateIndex
CREATE INDEX "clothing_items_category_idx" ON "clothing_items"("category");

-- CreateIndex
CREATE INDEX "outfit_recommendations_user_id_idx" ON "outfit_recommendations"("user_id");

-- CreateIndex
CREATE INDEX "outfit_recommendations_occasion_idx" ON "outfit_recommendations"("occasion");

-- CreateIndex
CREATE INDEX "saved_outfits_user_id_idx" ON "saved_outfits"("user_id");

-- AddForeignKey
ALTER TABLE "skin_tone_analysis" ADD CONSTRAINT "skin_tone_analysis_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "wardrobes" ADD CONSTRAINT "wardrobes_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "clothing_items" ADD CONSTRAINT "clothing_items_wardrobe_id_fkey" FOREIGN KEY ("wardrobe_id") REFERENCES "wardrobes"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "clothing_items" ADD CONSTRAINT "clothing_items_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "outfit_recommendations" ADD CONSTRAINT "outfit_recommendations_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "outfit_recommendations" ADD CONSTRAINT "outfit_recommendations_skin_tone_analysis_id_fkey" FOREIGN KEY ("skin_tone_analysis_id") REFERENCES "skin_tone_analysis"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "outfit_recommendations" ADD CONSTRAINT "outfit_recommendations_top_item_id_fkey" FOREIGN KEY ("top_item_id") REFERENCES "clothing_items"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "outfit_recommendations" ADD CONSTRAINT "outfit_recommendations_bottom_item_id_fkey" FOREIGN KEY ("bottom_item_id") REFERENCES "clothing_items"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "outfit_recommendations" ADD CONSTRAINT "outfit_recommendations_shoes_item_id_fkey" FOREIGN KEY ("shoes_item_id") REFERENCES "clothing_items"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "outfit_recommendations" ADD CONSTRAINT "outfit_recommendations_accessories_item_id_fkey" FOREIGN KEY ("accessories_item_id") REFERENCES "clothing_items"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "saved_outfits" ADD CONSTRAINT "saved_outfits_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "saved_outfits" ADD CONSTRAINT "saved_outfits_outfit_recommendation_id_fkey" FOREIGN KEY ("outfit_recommendation_id") REFERENCES "outfit_recommendations"("id") ON DELETE SET NULL ON UPDATE CASCADE;
