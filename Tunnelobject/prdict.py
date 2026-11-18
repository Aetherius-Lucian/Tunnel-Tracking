"""Compatibility entry point; use predict.py for new commands."""

if __package__:
    from .predict import main
else:
    from predict import main


if __name__ == "__main__":
    main()
