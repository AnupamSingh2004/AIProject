import { NextRequest, NextResponse } from 'next/server'
import prisma from '@/lib/prisma'

/**
 * POST /api/wardrobe/items
 * Add clothing item to wardrobe with image storage
 */
export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const file = formData.get('image') as File
    const wardrobeId = formData.get('wardrobeId') as string
    const userId = formData.get('userId') as string
    const name = formData.get('name') as string
    const category = formData.get('category') as string

    if (!file || !wardrobeId || !userId || !name || !category) {
      return NextResponse.json(
        { error: 'Missing required fields' },
        { status: 400 }
      )
    }

    // Find or create user
    let user = await prisma.user.findUnique({
      where: { id: userId },
    })

    if (!user) {
      console.log('User not found, creating new user')
      user = await prisma.user.create({
        data: {
          id: userId,
          email: `user-${userId}@styleai.demo`,
          name: 'Demo User',
        },
      })
    }

    // Convert image to buffer
    const bytes = await file.arrayBuffer()
    const buffer = Buffer.from(bytes)

    // Try to call AI backend to analyze clothing item (optional)
    let aiAnalysis = null
    try {
      const aiBackendUrl = process.env.AI_BACKEND_URL || 'http://localhost:8000'
      
      const aiFormData = new FormData()
      aiFormData.append('file', new Blob([buffer]), file.name)

      const aiResponse = await fetch(`${aiBackendUrl}/api/analyze-clothing`, {
        method: 'POST',
        body: aiFormData,
      })

      if (aiResponse.ok) {
        aiAnalysis = await aiResponse.json()
      }
    } catch (error) {
      console.log('AI backend not available, storing item without analysis')
    }

    // Store in database
    const clothingItem = await prisma.clothingItem.create({
      data: {
        wardrobeId,
        userId: user.id,
        name,
        image: buffer,
        imageFilename: file.name,
        imageMimetype: file.type,
        category,
        clothingType: aiAnalysis?.clothing_type,
        dominantColorR: aiAnalysis?.dominant_color?.r,
        dominantColorG: aiAnalysis?.dominant_color?.g,
        dominantColorB: aiAnalysis?.dominant_color?.b,
        secondaryColors: aiAnalysis?.secondary_colors || [],
        style: aiAnalysis?.style,
        pattern: aiAnalysis?.pattern,
        aiAnalyzed: !!aiAnalysis,
        aiConfidence: aiAnalysis?.confidence,
      },
    })

    return NextResponse.json({
      success: true,
      item: {
        id: clothingItem.id,
        name: clothingItem.name,
        category: clothingItem.category,
        dominantColor: {
          r: clothingItem.dominantColorR,
          g: clothingItem.dominantColorG,
          b: clothingItem.dominantColorB,
        },
        style: clothingItem.style,
        pattern: clothingItem.pattern,
      },
    })
  } catch (error) {
    console.error('Error adding clothing item:', error)
    return NextResponse.json(
      { error: 'Failed to add clothing item' },
      { status: 500 }
    )
  }
}

/**
 * GET /api/wardrobe/items?wardrobeId=xxx
 * Get all clothing items in a wardrobe
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const wardrobeId = searchParams.get('wardrobeId')
    const userId = searchParams.get('userId')

    if (!wardrobeId && !userId) {
      return NextResponse.json(
        { error: 'Wardrobe ID or User ID required' },
        { status: 400 }
      )
    }

    const where: any = {}
    
    if (wardrobeId) {
      where.wardrobeId = wardrobeId
    } else if (userId) {
      // Find user by ID and get their items
      const user = await prisma.user.findUnique({
        where: { id: userId },
      })
      
      if (user) {
        where.userId = user.id
      } else {
        return NextResponse.json({ items: [] })
      }
    }

    const items = await prisma.clothingItem.findMany({
      where,
      select: {
        id: true,
        name: true,
        category: true,
        clothingType: true,
        dominantColorR: true,
        dominantColorG: true,
        dominantColorB: true,
        style: true,
        pattern: true,
        season: true,
        occasion: true,
        createdAt: true,
        // Don't return image binary in list
      },
      orderBy: { createdAt: 'desc' },
    })

    return NextResponse.json({ items })
  } catch (error) {
    console.error('Error fetching clothing items:', error)
    return NextResponse.json(
      { error: 'Failed to fetch items' },
      { status: 500 }
    )
  }
}
