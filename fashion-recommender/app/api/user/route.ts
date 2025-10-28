import { NextRequest, NextResponse } from 'next/server'
import prisma from '@/lib/prisma'

/**
 * POST /api/user
 * Create or get a user
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { email, name } = body

    if (!email) {
      return NextResponse.json({ error: 'Email required' }, { status: 400 })
    }

    // Find or create user
    let user = await prisma.user.findUnique({
      where: { email },
    })

    if (!user) {
      user = await prisma.user.create({
        data: {
          email,
          name: name || null,
        },
      })
    }

    return NextResponse.json({ success: true, user })
  } catch (error) {
    console.error('Error creating/fetching user:', error)
    return NextResponse.json(
      { error: 'Failed to process user' },
      { status: 500 }
    )
  }
}

/**
 * GET /api/user?email=xxx or ?id=xxx
 * Get user by email or ID
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const email = searchParams.get('email')
    const id = searchParams.get('id')

    if (!email && !id) {
      return NextResponse.json(
        { error: 'Email or ID required' },
        { status: 400 }
      )
    }

    const user = await prisma.user.findUnique({
      where: email ? { email } : { id: id! },
      include: {
        _count: {
          select: {
            wardrobes: true,
            clothingItems: true,
            skinToneAnalyses: true,
            outfitRecommendations: true,
          },
        },
      },
    })

    if (!user) {
      return NextResponse.json({ error: 'User not found' }, { status: 404 })
    }

    return NextResponse.json({ user })
  } catch (error) {
    console.error('Error fetching user:', error)
    return NextResponse.json(
      { error: 'Failed to fetch user' },
      { status: 500 }
    )
  }
}

/**
 * PUT /api/user
 * Update user information
 */
export async function PUT(request: NextRequest) {
  try {
    const body = await request.json()
    const { id, name, email } = body

    if (!id) {
      return NextResponse.json({ error: 'User ID required' }, { status: 400 })
    }

    const user = await prisma.user.update({
      where: { id },
      data: {
        name: name !== undefined ? name : undefined,
        email: email || undefined,
      },
    })

    return NextResponse.json({ success: true, user })
  } catch (error) {
    console.error('Error updating user:', error)
    return NextResponse.json(
      { error: 'Failed to update user' },
      { status: 500 }
    )
  }
}
