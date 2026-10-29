param(
    [Parameter(Mandatory = $true)][string]$InputPng,
    [Parameter(Mandatory = $true)][string]$OutputPng,
    [Parameter(Mandatory = $true)][int]$TitleMaskX,
    [Parameter(Mandatory = $true)][int]$TitleMaskY,
    [Parameter(Mandatory = $true)][int]$TitleMaskWidth,
    [Parameter(Mandatory = $true)][int]$TitleMaskHeight,
    [Parameter(Mandatory = $true)][int]$TitleCenterX,
    [Parameter(Mandatory = $true)][int]$TitleCenterY1,
    [Parameter(Mandatory = $true)][int]$TitleCenterY2,
    [Parameter(Mandatory = $true)][int]$TitleFontPixels
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing

$source = $null
$bitmap = $null
$graphics = $null
$font = $null
$brush = $null
$format = $null

try {
    $source = [System.Drawing.Image]::FromFile((Resolve-Path -LiteralPath $InputPng).Path)
    $bitmap = New-Object System.Drawing.Bitmap(
        $source.Width,
        $source.Height,
        [System.Drawing.Imaging.PixelFormat]::Format24bppRgb
    )
    $bitmap.SetResolution(600, 600)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $graphics.Clear([System.Drawing.Color]::White)
    # Draw in explicit pixel coordinates.  DrawImageUnscaled can still honor
    # the source PNG's physical-resolution metadata in some GDI+ builds,
    # which enlarged the 600-dpi Origin export and cropped the right/bottom
    # of the panel.  Matching pixel-unit source and destination rectangles
    # preserves the Origin raster one-for-one outside the title mask.
    $destinationRectangle = New-Object System.Drawing.Rectangle(
        0,
        0,
        $source.Width,
        $source.Height
    )
    $graphics.DrawImage(
        $source,
        $destinationRectangle,
        0,
        0,
        $source.Width,
        $source.Height,
        [System.Drawing.GraphicsUnit]::Pixel
    )
    $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality
    $graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit

    $mask = New-Object System.Drawing.Rectangle(
        $TitleMaskX,
        $TitleMaskY,
        $TitleMaskWidth,
        $TitleMaskHeight
    )
    $graphics.FillRectangle([System.Drawing.Brushes]::White, $mask)

    $font = New-Object System.Drawing.Font(
        'Times New Roman',
        [single]$TitleFontPixels,
        [System.Drawing.FontStyle]::Regular,
        [System.Drawing.GraphicsUnit]::Pixel
    )
    $brush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(8, 8, 8))
    $format = New-Object System.Drawing.StringFormat
    $format.Alignment = [System.Drawing.StringAlignment]::Center
    $format.LineAlignment = [System.Drawing.StringAlignment]::Center

    $graphics.DrawString(
        'Grouped SHAP contribution',
        $font,
        $brush,
        (New-Object System.Drawing.PointF([single]$TitleCenterX, [single]$TitleCenterY1)),
        $format
    )
    $graphics.DrawString(
        'to predicted PCE (percentage points)',
        $font,
        $brush,
        (New-Object System.Drawing.PointF([single]$TitleCenterX, [single]$TitleCenterY2)),
        $format
    )

    $destination = [System.IO.Path]::GetFullPath($OutputPng)
    $directory = [System.IO.Path]::GetDirectoryName($destination)
    if (-not [System.IO.Directory]::Exists($directory)) {
        [System.IO.Directory]::CreateDirectory($directory) | Out-Null
    }
    $bitmap.Save($destination, [System.Drawing.Imaging.ImageFormat]::Png)
}
finally {
    if ($format) { $format.Dispose() }
    if ($brush) { $brush.Dispose() }
    if ($font) { $font.Dispose() }
    if ($graphics) { $graphics.Dispose() }
    if ($bitmap) { $bitmap.Dispose() }
    if ($source) { $source.Dispose() }
}
