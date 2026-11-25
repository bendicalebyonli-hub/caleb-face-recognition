Add-Type -AssemblyName System.Drawing

function Generate-BiometricIcon {
    param(
        [int]$size,
        [string]$outputPath
    )

    $bmp = New-Object System.Drawing.Bitmap($size, $size)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $g.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
    $g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic

    # Background gradient
    $rect = New-Object System.Drawing.Rectangle(0, 0, $size, $size)
    $bgBrush = New-Object System.Drawing.Drawing2D.LinearGradientBrush(
        $rect,
        [System.Drawing.ColorTranslator]::FromHtml('#0a0e17'),
        [System.Drawing.ColorTranslator]::FromHtml('#151d30'),
        45.0
    )
    $g.FillRectangle($bgBrush, $rect)

    # Outer rounded squircle
    $pad = [int]($size * 0.08)
    $innerSize = [int]($size - ($pad * 2))
    
    # Corner brackets (Face ID reticle)
    $bracketLen = [int]($innerSize * 0.22)
    $bracketPen = New-Object System.Drawing.Pen([System.Drawing.ColorTranslator]::FromHtml('#00e5ff'), [Math]::Max(3, [int]($size * 0.04)))
    $bracketPen.StartCap = [System.Drawing.Drawing2D.LineCap]::Round
    $bracketPen.EndCap = [System.Drawing.Drawing2D.LineCap]::Round

    # Top-Left
    $g.DrawLine($bracketPen, $pad, ($pad + $bracketLen), $pad, $pad)
    $g.DrawLine($bracketPen, $pad, $pad, ($pad + $bracketLen), $pad)

    # Top-Right
    $trX = [int]($size - $pad)
    $g.DrawLine($bracketPen, ($trX - $bracketLen), $pad, $trX, $pad)
    $g.DrawLine($bracketPen, $trX, $pad, $trX, ($pad + $bracketLen))

    # Bottom-Left
    $blY = [int]($size - $pad)
    $g.DrawLine($bracketPen, $pad, ($blY - $bracketLen), $pad, $blY)
    $g.DrawLine($bracketPen, $pad, $blY, ($pad + $bracketLen), $blY)

    # Bottom-Right
    $g.DrawLine($bracketPen, ($trX - $bracketLen), $blY, $trX, $blY)
    $g.DrawLine($bracketPen, $trX, ($blY - $bracketLen), $trX, $blY)

    # Biometric Face Silhouette in center
    $facePen = New-Object System.Drawing.Pen([System.Drawing.ColorTranslator]::FromHtml('#38bdf8'), [Math]::Max(2, [int]($size * 0.025)))
    $facePen.StartCap = [System.Drawing.Drawing2D.LineCap]::Round
    $facePen.EndCap = [System.Drawing.Drawing2D.LineCap]::Round

    $cx = [int]($size / 2)
    $cy = [int]($size / 2)
    $headW = [int]($size * 0.36)
    $headH = [int]($size * 0.44)

    # Draw Face Oval
    $faceX = [int]($cx - ($headW / 2))
    $faceY = [int]($cy - ($headH / 2))
    $g.DrawEllipse($facePen, $faceX, $faceY, $headW, $headH)

    # Eyes
    $eyeBrush = New-Object System.Drawing.SolidBrush([System.Drawing.ColorTranslator]::FromHtml('#00e5ff'))
    $eyeW = [Math]::Max(4, [int]($size * 0.045))
    $eyeSpacing = [int]($size * 0.09)
    $eyeY = [int]($cy - ($size * 0.06))
    $eye1X = [int]($cx - $eyeSpacing - ($eyeW / 2))
    $eye2X = [int]($cx + $eyeSpacing - ($eyeW / 2))
    $g.FillEllipse($eyeBrush, $eye1X, $eyeY, $eyeW, $eyeW)
    $g.FillEllipse($eyeBrush, $eye2X, $eyeY, $eyeW, $eyeW)

    # Smile / jawline contour
    $mouthPen = New-Object System.Drawing.Pen([System.Drawing.ColorTranslator]::FromHtml('#00e5ff'), [Math]::Max(2, [int]($size * 0.02)))
    $mouthPen.StartCap = [System.Drawing.Drawing2D.LineCap]::Round
    $mouthPen.EndCap = [System.Drawing.Drawing2D.LineCap]::Round
    $mouthY = [int]($cy + ($size * 0.08))
    $mouthW = [int]($size * 0.14)
    $mouthH = [int]($size * 0.08)
    $mouthX = [int]($cx - ($mouthW / 2))
    $g.DrawArc($mouthPen, $mouthX, [int]($mouthY - ($size * 0.04)), $mouthW, $mouthH, 20.0, 140.0)

    # Laser scan line across center
    $laserPen = New-Object System.Drawing.Pen([System.Drawing.ColorTranslator]::FromHtml('#10b981'), [Math]::Max(2, [int]($size * 0.018)))
    $laserPen.StartCap = [System.Drawing.Drawing2D.LineCap]::Round
    $laserPen.EndCap = [System.Drawing.Drawing2D.LineCap]::Round
    $lx1 = [int]($pad + ($innerSize * 0.15))
    $lx2 = [int]($size - $pad - ($innerSize * 0.15))
    $g.DrawLine($laserPen, $lx1, $cy, $lx2, $cy)

    # Glow dot on laser
    $laserDotBrush = New-Object System.Drawing.SolidBrush([System.Drawing.ColorTranslator]::FromHtml('#34d399'))
    $dotR = [Math]::Max(4, [int]($size * 0.035))
    $g.FillEllipse($laserDotBrush, [int]($cx - ($dotR/2)), [int]($cy - ($dotR/2)), $dotR, $dotR)

    # Save
    $bmp.Save($outputPath, [System.Drawing.Imaging.ImageFormat]::Png)
    $g.Dispose()
    $bmp.Dispose()
    Write-Output "Successfully generated: $outputPath"
}

$destDir = "C:\Users\USER\.gemini\antigravity\scratch\caleb-face-recognition-mobile-pwa\icons"
Generate-BiometricIcon -size 192 -outputPath "$destDir\icon-192.png"
Generate-BiometricIcon -size 512 -outputPath "$destDir\icon-512.png"
Generate-BiometricIcon -size 180 -outputPath "$destDir\apple-touch-icon.png"
Generate-BiometricIcon -size 512 -outputPath "$destDir\icon-maskable.png"
