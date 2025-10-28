import { NextRequest, NextResponse } from 'next/server'
import prisma from '@/lib/prisma'

/**
 * POST /api/saved-outfits
 * Save a favorite outfit combination
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { userId, name, outfitRecommendationId, notes } = body

    if (!userId || !name) {
      return NextResponse.json(
        { error: 'User ID and name required' },
        { status: 400 }
      )
    }

    const savedOutfit = await prisma.savedOutfit.create({
      data: {
        userId,
        name,
        outfitRecommendationId: outfitRecommendationId || null,
        notes: notes || null,
      },
      include: {
        outfitRecommendation: true,
      },
    })

    return NextResponse.json({ success: true, savedOutfit })
  } catch (error) {
    console.error('Error saving outfit:', error)
    return NextResponse.json(
      { error: 'Failed to save outfit' },
      { status: 500 }
    )
  }
}

/**
 * GET /api/saved-outfits?userId=xxx
 * Get all saved outfits for a user
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const userId = searchParams.get('userId')

    if (!userId) {
      return NextResponse.json({ error: 'User ID required' }, { status: 400 })
    }

    const savedOutfits = await prisma.savedOutfit.findMany({
      where: { userId },
      include: {
        outfitRecommendation: true,
      },
      orderBy: { createdAt: 'desc' },
    })

    return NextResponse.json({ savedOutfits })
  } catch (error) {
    console.error('Error fetching saved outfits:', error)
    return NextResponse.json(
      { error: 'Failed to fetch saved outfits' },
      { status: 500 }
    )
  }
}

/**
 * DELETE /api/saved-outfits?id=xxx
 * Delete a saved outfit
 */
export async function DELETE(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const id = searchParams.get('id')

    if (!id) {
      return NextResponse.json(
        { error: 'Outfit ID required' },
        { status: 400 }
      )
    }

    await prisma.savedOutfit.delete({
      where: { id },
    })

    return NextResponse.json({ success: true })
  } catch (error) {
    console.error('Error deleting saved outfit:', error)
    return NextResponse.json(
      { error: 'Failed to delete outfit' },
      { status: 500 }
    )
  }
}
