import { NextRequest, NextResponse } from 'next/server'
import prisma from '@/lib/prisma'

/**
 * POST /api/skin-analysis
 * Analyzes skin tone from uploaded photo
 */
export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const file = formData.get('photo') as File
    const userId = formData.get('userId') as string

    if (!file) {
      return NextResponse.json({ error: 'No photo provided' }, { status: 400 })
    }

    if (!userId) {
      return NextResponse.json({ error: 'User ID required' }, { status: 400 })
    }

    // Convert file to buffer for storage
    const bytes = await file.arrayBuffer()
    const buffer = Buffer.from(bytes)

    // Call Python AI backend for analysis
    const aiBackendUrl = process.env.AI_BACKEND_URL || 'http://localhost:8000'
    
    const aiFormData = new FormData()
    aiFormData.append('file', new Blob([buffer]), file.name)

    const aiResponse = await fetch(`${aiBackendUrl}/api/analyze-skin-tone`, {
      method: 'POST',
      body: aiFormData,
    })

    if (!aiResponse.ok) {
      throw new Error('AI analysis failed')
    }

    const analysisResult = await aiResponse.json()

    // Store in database with image
    const skinToneAnalysis = await prisma.skinToneAnalysis.create({
      data: {
        userId,
        photo: buffer,
        photoFilename: file.name,
        fitzpatrickType: analysisResult.fitzpatrick_type,
        undertone: analysisResult.undertone,
        dominantColorR: analysisResult.dominant_color.r,
        dominantColorG: analysisResult.dominant_color.g,
        dominantColorB: analysisResult.dominant_color.b,
        confidence: analysisResult.confidence,
      },
    })

    return NextResponse.json({
      success: true,
      analysis: {
        id: skinToneAnalysis.id,
        fitzpatrickType: skinToneAnalysis.fitzpatrickType,
        undertone: skinToneAnalysis.undertone,
        dominantColor: {
          r: skinToneAnalysis.dominantColorR,
          g: skinToneAnalysis.dominantColorG,
          b: skinToneAnalysis.dominantColorB,
        },
        confidence: skinToneAnalysis.confidence,
      },
    })
  } catch (error) {
    console.error('Skin analysis error:', error)
    return NextResponse.json(
      { error: 'Failed to analyze skin tone' },
      { status: 500 }
    )
  }
}

/**
 * GET /api/skin-analysis?userId=xxx
 * Get user's skin tone analysis history
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const userId = searchParams.get('userId')

    if (!userId) {
      return NextResponse.json({ error: 'User ID required' }, { status: 400 })
    }

    const analyses = await prisma.skinToneAnalysis.findMany({
      where: { userId },
      orderBy: { analyzedAt: 'desc' },
      select: {
        id: true,
        fitzpatrickType: true,
        undertone: true,
        dominantColorR: true,
        dominantColorG: true,
        dominantColorB: true,
        confidence: true,
        analyzedAt: true,
        // Don't return photo binary data in list
      },
    })

    return NextResponse.json({ analyses })
  } catch (error) {
    console.error('Error fetching skin analyses:', error)
    return NextResponse.json(
      { error: 'Failed to fetch analyses' },
      { status: 500 }
    )
  }
}
