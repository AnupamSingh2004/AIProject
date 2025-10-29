import { NextRequest, NextResponse } from 'next/server';
import { prisma } from '@/lib/prisma';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { userId, skinType, undertone, dominantColor, confidence } = body;

    console.log('Received skin tone save request:', { userId, skinType, undertone, dominantColor, confidence });

    // Validate required fields
    if (!userId || !skinType || !undertone || !dominantColor) {
      console.error('Missing required fields:', { userId, skinType, undertone, dominantColor });
      return NextResponse.json(
        { error: 'Missing required fields' },
        { status: 400 }
      );
    }

    // Parse hex color to RGB
    const hex = dominantColor.replace('#', '');
    const r = parseInt(hex.substring(0, 2), 16);
    const g = parseInt(hex.substring(2, 4), 16);
    const b = parseInt(hex.substring(4, 6), 16);

    console.log('Parsed RGB values:', { r, g, b });

    // Map skin type to Fitzpatrick type
    const fitzpatrickType = skinType.includes('IV') ? 'IV' : 
                           skinType.includes('III') ? 'III' :
                           skinType.includes('II') ? 'II' :
                           skinType.includes('I') ? 'I' :
                           skinType.includes('V') ? 'V' :
                           skinType.includes('VI') ? 'VI' : 'IV';

    console.log('Mapped Fitzpatrick type:', fitzpatrickType);

    // Create a dummy photo (1x1 pixel transparent PNG)
    const dummyPhoto = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==', 'base64');

    // Ensure user exists in database
    let user = await prisma.user.findUnique({
      where: { id: userId },
    });

    if (!user) {
      console.log('User not found, creating new user');
      user = await prisma.user.create({
        data: {
          id: userId,
          email: `user-${userId}@styleai.demo`,
          name: 'Demo User',
        },
      });
    }

    // Check if user already has a skin tone analysis
    const existingAnalysis = await prisma.skinToneAnalysis.findFirst({
      where: { userId },
    });

    console.log('Existing analysis:', existingAnalysis ? 'Found' : 'Not found');

    let analysis;
    if (existingAnalysis) {
      // Update existing analysis
      analysis = await prisma.skinToneAnalysis.update({
        where: { id: existingAnalysis.id },
        data: {
          fitzpatrickType,
          undertone,
          dominantColorR: r,
          dominantColorG: g,
          dominantColorB: b,
          confidence: confidence ? confidence / 100 : 0.92,
        },
      });
      console.log('Updated existing analysis');
    } else {
      // Create new analysis
      analysis = await prisma.skinToneAnalysis.create({
        data: {
          userId,
          photo: dummyPhoto,
          photoFilename: 'analysis.png',
          fitzpatrickType,
          undertone,
          dominantColorR: r,
          dominantColorG: g,
          dominantColorB: b,
          confidence: confidence ? confidence / 100 : 0.92,
        },
      });
      console.log('Created new analysis');
    }

    return NextResponse.json({
      success: true,
      analysis,
      message: 'Skin tone analysis saved successfully',
    });
  } catch (error) {
    console.error('Error saving skin tone analysis:', error);
    if (error instanceof Error) {
      console.error('Error stack:', error.stack);
    }
    return NextResponse.json(
      { error: `Failed to save skin tone analysis: ${error instanceof Error ? error.message : 'Unknown error'}` },
      { status: 500 }
    );
  }
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const userId = searchParams.get('userId');

    if (!userId) {
      return NextResponse.json(
        { error: 'userId is required' },
        { status: 400 }
      );
    }

    const analysis = await prisma.skinToneAnalysis.findUnique({
      where: { userId },
    });

    if (!analysis) {
      return NextResponse.json(
        { error: 'No skin tone analysis found for this user' },
        { status: 404 }
      );
    }

    return NextResponse.json(analysis);
  } catch (error) {
    console.error('Error fetching skin tone analysis:', error);
    return NextResponse.json(
      { error: 'Failed to fetch skin tone analysis' },
      { status: 500 }
    );
  }
}
