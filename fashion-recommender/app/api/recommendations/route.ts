import { NextRequest, NextResponse } from 'next/server'
import prisma from '@/lib/prisma'
import { ClothingItem } from '@prisma/client'

interface AIOutfit {
  top_id?: string | null
  bottom_id?: string | null
  shoes_id?: string | null
  accessories_id?: string | null
  compatibility_score: number
  skin_tone_match_score?: number | null
  color_harmony_type?: string | null
  reason?: string | null
}

/**
 * POST /api/recommendations
 * Generate outfit recommendations based on skin tone, wardrobe, and preferences
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { userId, skinToneAnalysisId, occasion, season, count = 10 } = body

    if (!userId) {
      return NextResponse.json({ error: 'User ID required' }, { status: 400 })
    }

    // Get user's skin tone analysis
    let skinTone = null
    if (skinToneAnalysisId) {
      skinTone = await prisma.skinToneAnalysis.findUnique({
        where: { id: skinToneAnalysisId },
      })
    }

    // Get user's wardrobe items
    const clothingItems = await prisma.clothingItem.findMany({
      where: { userId },
    })

    if (clothingItems.length === 0) {
      return NextResponse.json(
        { error: 'No clothing items in wardrobe' },
        { status: 400 }
      )
    }

    // Call AI backend to generate recommendations
    const aiBackendUrl = process.env.AI_BACKEND_URL || 'http://localhost:8000'
    
    const aiResponse = await fetch(`${aiBackendUrl}/api/recommend-outfits`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        skinTone: skinTone ? {
          fitzpatrick_type: skinTone.fitzpatrickType,
          undertone: skinTone.undertone,
          dominant_color: {
            r: skinTone.dominantColorR,
            g: skinTone.dominantColorG,
            b: skinTone.dominantColorB,
          },
        } : null,
        wardrobe: clothingItems.map((item: ClothingItem) => ({
          id: item.id,
          category: item.category,
          dominant_color: {
            r: item.dominantColorR,
            g: item.dominantColorG,
            b: item.dominantColorB,
          },
          style: item.style,
          pattern: item.pattern,
        })),
        occasion,
        season,
        count,
      }),
    })

    if (!aiResponse.ok) {
      throw new Error('AI recommendation failed')
    }

    const recommendations = await aiResponse.json()

    // Save recommendations to database
    const savedRecommendations = await Promise.all(
      recommendations.outfits.map(async (outfit: AIOutfit) => {
        return await prisma.outfitRecommendation.create({
          data: {
            userId,
            skinToneAnalysisId: skinToneAnalysisId || null,
            topItemId: outfit.top_id || null,
            bottomItemId: outfit.bottom_id || null,
            shoesItemId: outfit.shoes_id || null,
            accessoriesItemId: outfit.accessories_id || null,
            occasion: occasion || 'Casual',
            season: season || null,
            compatibilityScore: outfit.compatibility_score,
            skinToneMatchScore: outfit.skin_tone_match_score,
            colorHarmonyType: outfit.color_harmony_type,
            reason: outfit.reason,
          },
        })
      })
    )

    // Fetch full outfit details with item information
    const outfitsWithDetails = await Promise.all(
      savedRecommendations.map(async (rec) => {
        return await prisma.outfitRecommendation.findUnique({
          where: { id: rec.id },
          include: {
            topItem: true,
            bottomItem: true,
            shoesItem: true,
            accessoriesItem: true,
          },
        })
      })
    )

    return NextResponse.json({
      success: true,
      recommendations: outfitsWithDetails,
    })
  } catch (error) {
    console.error('Error generating recommendations:', error)
    return NextResponse.json(
      { error: 'Failed to generate recommendations' },
      { status: 500 }
    )
  }
}

/**
 * GET /api/recommendations?userId=xxx
 * Get user's recommendation history
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const userId = searchParams.get('userId')

    if (!userId) {
      return NextResponse.json({ error: 'User ID required' }, { status: 400 })
    }

    const recommendations = await prisma.outfitRecommendation.findMany({
      where: { userId },
      include: {
        topItem: {
          select: {
            id: true,
            name: true,
            category: true,
            dominantColorR: true,
            dominantColorG: true,
            dominantColorB: true,
          },
        },
        bottomItem: {
          select: {
            id: true,
            name: true,
            category: true,
            dominantColorR: true,
            dominantColorG: true,
            dominantColorB: true,
          },
        },
        shoesItem: {
          select: {
            id: true,
            name: true,
            category: true,
            dominantColorR: true,
            dominantColorG: true,
            dominantColorB: true,
          },
        },
      },
      orderBy: { createdAt: 'desc' },
      take: 50,
    })

    return NextResponse.json({ recommendations })
  } catch (error) {
    console.error('Error fetching recommendations:', error)
    return NextResponse.json(
      { error: 'Failed to fetch recommendations' },
      { status: 500 }
    )
  }
}
