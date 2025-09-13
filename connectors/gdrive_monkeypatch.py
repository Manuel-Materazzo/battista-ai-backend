import wrapt
import pathway.io.gdrive


@wrapt.patch_function_wrapper('pathway.io.gdrive', 'extend_metadata')
def extend_metadata_wrapper(wrapped, instance, args, kwargs):
    """Enhance google drive metadata by fixing paths for files in folders.

    This wrapper modifies the metadata returned by the original extend_metadata function
    to properly handle paths for files that are located in folders. When a file has
    parent folders, the path is updated to include the parent folder structure.

    Args:
        wrapped: The original function being wrapped.
        instance: The instance of the class (if applicable).
        args: Positional arguments passed to the original function.
        kwargs: Keyword arguments passed to the original function.

    Returns:
        dict: The enhanced metadata with updated path if necessary.
    """
    # Call the original function
    metadata = wrapped(*args, **kwargs)

    # Fix path for files in folders
    if len(metadata.get('parents', [])) > 0 and 'path' in metadata:
        metadata["path"] = "/".join(metadata["parents"]) + "/" + metadata["name"]

    return metadata
