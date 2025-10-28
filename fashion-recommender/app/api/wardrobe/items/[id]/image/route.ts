import { NextRequest, NextResponse } from 'next/server'
import prisma from '@/lib/prisma'

/**
 * GET /api/wardrobe/items/[id]/image
 * Retrieve image for a specific clothing item
 */
export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const clothingItem = await prisma.clothingItem.findUnique({
      where: { id: params.id },
      select: {
        image: true,
        imageMimetype: true,
        imageFilename: true,
      },
    })

    if (!clothingItem || !clothingItem.image) {
      return NextResponse.json({ error: 'Image not found' }, { status: 404 })
    }

    // Return image as binary data
    return new NextResponse(clothingItem.image, {
      headers: {
        'Content-Type': clothingItem.imageMimetype || 'image/jpeg',
        'Content-Disposition': `inline; filename="${clothingItem.imageFilename}"`,
      },
    })
  } catch (error) {
    console.error('Error fetching image:', error)
    return NextResponse.json(
      { error: 'Failed to fetch image' },
      { status: 500 }
    )
  }
}
