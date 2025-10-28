import { NextRequest, NextResponse } from 'next/server'
import prisma from '@/lib/prisma'

/**
 * POST /api/wardrobe
 * Create a new wardrobe
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { userId, name, description } = body

    if (!userId || !name) {
      return NextResponse.json(
        { error: 'User ID and name required' },
        { status: 400 }
      )
    }

    // Check if user exists, create if not
    let user = await prisma.user.findUnique({
      where: { email: userId },
    })

    if (!user) {
      user = await prisma.user.create({
        data: {
          email: userId,
          name: 'Demo User',
        },
      })
    }

    // Check if wardrobe exists
    let wardrobe = await prisma.wardrobe.findFirst({
      where: { userId: user.id },
    })

    if (!wardrobe) {
      wardrobe = await prisma.wardrobe.create({
        data: {
          userId: user.id,
          name,
          description: description || null,
        },
      })
    }

    return NextResponse.json({ success: true, wardrobe })
  } catch (error) {
    console.error('Error creating wardrobe:', error)
    return NextResponse.json(
      { error: 'Failed to create wardrobe' },
      { status: 500 }
    )
  }
}

/**
 * GET /api/wardrobe?userId=xxx
 * Get all wardrobes for a user
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const userId = searchParams.get('userId')

    if (!userId) {
      return NextResponse.json({ error: 'User ID required' }, { status: 400 })
    }

    // Find user by email
    const user = await prisma.user.findUnique({
      where: { email: userId },
    })

    if (!user) {
      return NextResponse.json({ wardrobes: [] })
    }

    const wardrobes = await prisma.wardrobe.findMany({
      where: { userId: user.id },
      include: {
        _count: {
          select: { clothingItems: true },
        },
      },
      orderBy: { createdAt: 'desc' },
    })

    return NextResponse.json({ wardrobes })
  } catch (error) {
    console.error('Error fetching wardrobes:', error)
    return NextResponse.json(
      { error: 'Failed to fetch wardrobes' },
      { status: 500 }
    )
  }
}

/**
 * PUT /api/wardrobe
 * Update a wardrobe
 */
export async function PUT(request: NextRequest) {
  try {
    const body = await request.json()
    const { id, name, description } = body

    if (!id) {
      return NextResponse.json(
        { error: 'Wardrobe ID required' },
        { status: 400 }
      )
    }

    const wardrobe = await prisma.wardrobe.update({
      where: { id },
      data: {
        name: name || undefined,
        description: description !== undefined ? description : undefined,
      },
    })

    return NextResponse.json({ success: true, wardrobe })
  } catch (error) {
    console.error('Error updating wardrobe:', error)
    return NextResponse.json(
      { error: 'Failed to update wardrobe' },
      { status: 500 }
    )
  }
}

/**
 * DELETE /api/wardrobe?id=xxx
 * Delete a wardrobe
 */
export async function DELETE(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const id = searchParams.get('id')

    if (!id) {
      return NextResponse.json(
        { error: 'Wardrobe ID required' },
        { status: 400 }
      )
    }

    await prisma.wardrobe.delete({
      where: { id },
    })

    return NextResponse.json({ success: true })
  } catch (error) {
    console.error('Error deleting wardrobe:', error)
    return NextResponse.json(
      { error: 'Failed to delete wardrobe' },
      { status: 500 }
    )
  }
}
