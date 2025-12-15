import click

from logic.image_processor import (
    get_available_classes,
    predict_class,
    preprocess_image,
    resize_image,
)


@click.group()
@click.version_option(version="1.0.0")
def app():
    pass


@app.command()
@click.argument("image_path", type=click.Path(exists=True))
def predict(image_path):
    try:
        predicted_breed, confidence = predict_class(image_path)
        click.echo(f"Predicted pet breed: {predicted_breed}")
        click.echo(f"Confidence: {confidence:.2%}")
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()


@app.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--width", "-w", type=int, required=True, help="Target width")
@click.option("--height", "-h", type=int, required=True, help="Target height")
def resize(image_path, output_path, width, height):
    try:
        new_size = resize_image(image_path, output_path, (width, height))
        click.echo(f"Image resized to {new_size[0]}x{new_size[1]} and saved to {output_path}")
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()


@app.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--normalize/--no-normalize", default=True, help="Normalize pixel values")
@click.option("--grayscale/--no-grayscale", default=False, help="Convert to grayscale")
def preprocess(image_path, output_path, normalize, grayscale):
    try:
        result = preprocess_image(image_path, output_path, normalize, grayscale)
        click.echo("Image preprocessed successfully:")
        click.echo(f"  Original size: {result['original_size']}")
        click.echo(f"  Final size: {result['final_size']}")
        click.echo(f"  Mode: {result['mode']}")
        click.echo(f"  Normalized: {result['normalized']}")
        click.echo(f"  Grayscale: {result['grayscale']}")
        click.echo(f"  Saved to: {output_path}")
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()


@app.command()
def classes():
    breeds = get_available_classes()
    click.echo("Available pet breeds:")
    for i, breed in enumerate(breeds, 1):
        click.echo(f"  {i}. {breed}")


if __name__ == "__main__":
    app()
