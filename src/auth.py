#!/usr/bin/env python3

"""
OneDrive backup integration for 2D Point Annotator.
Provides authentication and file upload to SharePoint/OneDrive.

Usage:
    from auth import OneDriveBackup

    backup = OneDriveBackup()
    backup.upload_backup("/path/to/file.db")  # Uploads to user/date folder
"""

from __future__ import annotations

import asyncio
import getpass
import logging
import os
import socket
import threading
import time
import tkinter as tk

# Prioritize IPv4 to avoid 5s hangs on broken IPv6 routes (common on some networks)
_original_getaddrinfo = socket.getaddrinfo


def _getaddrinfo_prioritize_ipv4(*args, **kwargs):
    responses = _original_getaddrinfo(*args, **kwargs)
    # Sort to put AF_INET (IPv4) first, keeping all responses for fallback
    return sorted(responses, key=lambda r: r[0] != socket.AF_INET)


socket.getaddrinfo = _getaddrinfo_prioritize_ipv4
import webbrowser
from datetime import datetime
from pathlib import Path
from tkinter import font, ttk
from typing import Callable, Optional, Sequence

from azure.identity import (
    AuthenticationRecord,
    DeviceCodeCredential,
    TokenCachePersistenceOptions,
)
from msgraph import GraphServiceClient
from msgraph.generated.models.drive_item import DriveItem
from msgraph.generated.models.folder import Folder

from dirs import AUTH_DIR

logger = logging.getLogger(__name__)

# Azure AD app registration
CLIENT_ID = "d6a282f7-2887-4e76-b92c-e8ca63ea5c8b"
TENANT_ID = "f59bcc9c-bb13-4ca4-a6ec-ac126b6ec4b9"
SCOPES = ["Files.ReadWrite.All"]

# SharePoint drive ID for backup storage
SHAREPOINT_DRIVE_ID = (
    "b!KMPrFjP2-EyF3_IAVjRxqwRrQzp7wJ5Bj5GN_uL5GeVd5oQt5J70S4Ie1tl8qDeE"
)
BASE_BACKUP_FOLDER = "pelvic-2d-points-backup"

AUTH_RECORD_PATH = AUTH_DIR / "auth_record.json"

# Module-level cached credential to avoid MSAL token-cache file-lock
# contention when many DeviceCodeCredential instances are created rapidly.
_shared_credential: Optional[DeviceCodeCredential] = None
_credential_lock = threading.Lock()

# Global reference to auth dialog (to prevent garbage collection)
_auth_dialog: Optional[tk.Toplevel] = None


def _show_auth_dialog(uri: str, code: str, done_event: Optional[threading.Event] = None) -> None:
    """Show a Tkinter dialog with the authentication code.

    Parameters
    ----------
    uri : str
        The device code login URI.
    code : str
        The device code to display.
    done_event : threading.Event, optional
        If provided, ``set()`` is called when the dialog is closed or when
        the user dismisses it.  Used by ``_threadsafe_prompt_callback`` to
        synchronise with the calling thread.
    """
    global _auth_dialog

    # Try to get existing Tk root, or create temporary one
    try:
        # Access the default root (private attribute)
        root = getattr(tk, "_default_root", None)
        if root is None:
            # No Tk root exists yet - create a temporary hidden one
            root = tk.Tk()
            root.withdraw()
    except Exception:
        root = tk.Tk()
        root.withdraw()

    # Create dialog window
    dialog = tk.Toplevel(root)
    dialog.title("OneDrive Login Required")
    dialog.geometry("500x350")
    dialog.resizable(False, False)

    # Center on screen
    dialog.update_idletasks()
    x = (dialog.winfo_screenwidth() - 500) // 2
    y = (dialog.winfo_screenheight() - 350) // 2
    dialog.geometry(f"500x350+{x}+{y}")

    # Make it stay on top
    dialog.attributes("-topmost", True)
    dialog.lift()
    dialog.focus_force()

    # Main frame with padding
    frame = ttk.Frame(dialog, padding=20)
    frame.pack(fill="both", expand=True)

    # Title
    title_label = ttk.Label(
        frame,
        text="ONE-TIME LOGIN",
        font=("Helvetica", 18, "bold"),
    )
    title_label.pack(pady=(0, 15))

    instructions_font = font.nametofont("TkDefaultFont").copy()
    instructions_font.configure(size=28)

    # Instructions
    instructions = ttk.Label(
        frame,
        text="A browser window has opened.\nEnter this code to sign in:",
        font=instructions_font,
        justify="center",
    )
    instructions.pack(pady=(0, 20))

    # Code display (large, prominent)
    code_frame = ttk.Frame(frame)
    code_frame.pack(pady=10)

    code_font = font.nametofont("TkDefaultFont").copy()
    code_font.configure(size=36)

    code_label = ttk.Label(
        code_frame,
        text=code,
        font=code_font,
        foreground="#0066cc",
    )
    code_label.pack(padx=20, pady=15)

    # Copy button
    def copy_code():
        dialog.clipboard_clear()
        dialog.clipboard_append(code)
        copy_btn.config(text="Copied!")
        dialog.after(1500, lambda: copy_btn.config(text="Copy Code"))

    copy_btn = ttk.Button(
        frame,
        text="Copy Code",
        command=copy_code,
    )
    copy_btn.pack(pady=10)

    # URL info (smaller)
    url_label = ttk.Label(
        frame,
        text=f"URL: {uri}",
        font=("Helvetica", 9),
        foreground="gray",
    )
    url_label.pack(pady=(15, 0))

    # Note about closing
    note_label = ttk.Label(
        frame,
        text="This window will close automatically after login.",
        font=("Helvetica", 9, "italic"),
        foreground="gray",
    )
    note_label.pack(pady=(5, 0))

    # When the dialog is destroyed (e.g. after login completes), signal
    # the waiting thread so it unblocks.
    if done_event is not None:
        def _on_dialog_destroy(_event=None):
            done_event.set()

        dialog.bind("<Destroy>", _on_dialog_destroy)

    # Store reference globally to prevent garbage collection
    _auth_dialog = dialog

    # Force update
    dialog.update()


def _close_auth_dialog() -> None:
    """Close the auth dialog if it exists."""
    global _auth_dialog
    if _auth_dialog is not None:
        try:
            _auth_dialog.destroy()
        except Exception:
            pass
        _auth_dialog = None


def _console_prompt_callback(uri: str, code: str, expires_on) -> None:
    """Fallback prompt callback that uses console output.

    Used when no Tk root is available (headless / no display).
    """
    if uri:
        try:
            webbrowser.open(uri, new=2)
        except Exception:
            pass

    print(f"\n{'=' * 60}")
    print("ONE-TIME LOGIN")
    print(f"Open: {uri}")
    print(f"\nENTER THIS CODE:\n\n    {code}\n")
    print("=" * 60)


def _prompt_callback(uri: str, code: str, expires_on) -> None:
    """Display device code authentication prompt via Tkinter dialog."""
    # Open browser
    if uri:
        try:
            webbrowser.open(uri, new=2)
        except Exception:
            pass

    # Show Tkinter dialog with code
    try:
        _show_auth_dialog(uri, code)
    except Exception as e:
        # Fallback to console if Tkinter fails
        logger.warning(f"Failed to show auth dialog: {e}")
        _console_prompt_callback(uri, code, expires_on)


def _threadsafe_prompt_callback(
    tk_root: tk.Misc, original_callback: Callable
) -> Callable:
    """Factory that returns a prompt callback safe to call from any thread.

    The returned callback marshals dialog creation to the main Tk thread
    via ``tk_root.after(0, ...)`` and blocks the calling (background) thread
    until the dialog is closed.  If the main-thread dispatch fails, it falls
    back to ``original_callback`` directly.

    Parameters
    ----------
    tk_root : tk.Misc
        The Tk root window (must be running a mainloop on the main thread).
    original_callback : callable
        The original ``prompt_callback(uri, code, expires_on)`` to fall
        back to if marshalling fails.

    Returns
    -------
    callable
        A wrapper with the same ``(uri, code, expires_on)`` signature.
    """

    def _wrapper(uri: str, code: str, expires_on) -> None:
        done_event = threading.Event()

        try:
            # Marshal the dialog creation to the main thread
            tk_root.after(0, lambda: _show_auth_dialog(uri, code, done_event=done_event))
        except Exception as e:
            logger.warning("Failed to marshal auth dialog to main thread: %s", e)
            # Fall back to original callback (console or direct)
            original_callback(uri, code, expires_on)
            return

        # Open browser from this thread (safe — no Tk interaction)
        if uri:
            try:
                webbrowser.open(uri, new=2)
            except Exception:
                pass

        # Block until the dialog signals completion
        done_event.wait()

    return _wrapper


def get_safe_username() -> str:
    """Get username for folder paths. Cross-platform safe."""
    try:
        return getpass.getuser()
    except (KeyError, OSError):
        import os

        return os.getenv("USER") or os.getenv("USERNAME") or "default_user"


def get_date_folder() -> str:
    """Get today's date as folder name (YYYY-MM-DD)."""
    return datetime.now().strftime("%Y-%m-%d")


def _get_or_create_credential(
    prompt_callback_fn=None,
) -> DeviceCodeCredential:
    """
    Get or create the module-level cached DeviceCodeCredential.

    MSAL's persistent token cache (``onedrive-ml``) uses file-based locking.
    Creating many DeviceCodeCredential instances rapidly causes lock contention
    and hangs.  We keep a single process-wide credential and reuse it.

    Parameters
    ----------
    prompt_callback_fn : callable, optional
        Callback for device-code prompts.  Only used when a *new* credential
        is created (i.e., the first call).  Subsequent calls return the
        cached credential regardless of this parameter.
    """
    global _shared_credential

    if _shared_credential is not None:
        return _shared_credential

    with _credential_lock:
        if _shared_credential is not None:
            return _shared_credential

        # Load cached auth record if exists
        record = None
        if AUTH_RECORD_PATH.exists():
            try:
                record = AuthenticationRecord.deserialize(AUTH_RECORD_PATH.read_text())
                logger.info("Loaded existing auth record")
            except Exception as e:
                logger.warning("Failed to load auth record: %s", e)
                record = None

        # Setup credential with token cache
        cache_opts = TokenCachePersistenceOptions(
            name="onedrive-ml", allow_unencrypted_storage=True
        )

        _shared_credential = DeviceCodeCredential(
            client_id=CLIENT_ID,
            tenant_id=TENANT_ID,
            cache_persistence_options=cache_opts,
            authentication_record=record,
            prompt_callback=prompt_callback_fn or _prompt_callback,
        )
        logger.info("Module-level credential created")
        return _shared_credential


def get_graph_client(
    prompt_callback_fn=None, tk_root: Optional[tk.Misc] = None
) -> GraphServiceClient:
    """
    Create an authenticated Graph client, triggering device-code auth if needed.

    Reuses cached credentials from AUTH_RECORD_PATH. If no cached credentials
    exist or they've expired beyond refresh, triggers the device-code flow.

    Parameters
    ----------
    prompt_callback_fn : callable, optional
        Custom callback for device-code prompts. Defaults to the module's
        _prompt_callback (which shows the Tkinter dialog).

    Returns
    -------
    GraphServiceClient
        An authenticated client ready for API calls.

    Raises
    ------
    Exception
        If authentication fails completely.
    """
    # Resolve the effective prompt callback — prefer a thread-safe wrapper
    # when a Tk root is available so the device-code dialog is always created
    # on the main thread.
    effective_callback = prompt_callback_fn
    if effective_callback is None:
        effective_callback = _prompt_callback
    if tk_root is not None:
        effective_callback = _threadsafe_prompt_callback(tk_root, effective_callback)

    credential = _get_or_create_credential(effective_callback)

    # Always try to get a token to verify/refresh credentials
    try:
        logger.info("Verifying/refreshing token...")
        token = credential.get_token(*SCOPES)
        logger.info("Token acquired (expires: %s)", token.expires_on)

        # If we didn't have a record before, save one now
        if not AUTH_RECORD_PATH.exists():
            logger.info("No auth record existed, creating one...")
            record = credential.authenticate(scopes=SCOPES)
            _save_auth_record(record)
            _close_auth_dialog()
            logger.info("Auth record saved")

    except Exception as token_error:
        logger.warning("Token acquisition failed: %s", token_error)
        logger.info("Re-authenticating with device code...")
        record = credential.authenticate(scopes=SCOPES)
        _save_auth_record(record)
        _close_auth_dialog()
        logger.info("Re-authentication successful, saved new auth record")

    client = GraphServiceClient(credentials=credential, scopes=SCOPES)
    logger.info("Client created successfully")
    return client


def _save_auth_record(record: AuthenticationRecord) -> None:
    """Atomically persist an AuthenticationRecord with restrictive permissions.

    Writes to a temporary file first, then atomically renames it into place
    to avoid corruption from partial writes.  Sets file permissions to 0o600
    (owner read/write only) for security.
    """
    AUTH_RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = AUTH_RECORD_PATH.with_suffix(".tmp")
    tmp_path.write_text(record.serialize())
    # Atomic rename — on POSIX this is guaranteed by the OS
    tmp_path.replace(AUTH_RECORD_PATH)
    # Restrict permissions: owner read/write only
    try:
        os.chmod(str(AUTH_RECORD_PATH), 0o600)
    except OSError as e:
        logger.warning("Failed to set auth record permissions: %s", e)


class OneDriveBackup:
    """
    Manages OneDrive backup operations for annotation files.

    Uploads to: pelvic-2d-points-backup/<username>/<YYYY-MM-DD>/

    Thread-safe: upload operations run in background threads.
    """

    def __init__(self) -> None:
        self._client: Optional[GraphServiceClient] = None
        self._credential: Optional[DeviceCodeCredential] = None
        self._lock = threading.Lock()
        self._initialized = False

    def _ensure_initialized(self) -> bool:
        """Lazy initialization of Graph client. Returns True if ready."""
        if self._initialized:
            return True

        with self._lock:
            if self._initialized:
                return True

            try:
                self._client = get_graph_client()
                self._credential = None  # credential is managed by get_graph_client
                self._initialized = True
                logger.info("Initialization complete")
                return True

            except Exception as e:
                logger.error("Failed to initialize OneDrive client: %s", e)
                return False

    def _get_shared_credential(self) -> Optional[DeviceCodeCredential]:
        """
        Get the module-level shared credential for token acquisition.

        Delegates to ``_get_or_create_credential`` which maintains a single
        process-wide DeviceCodeCredential to avoid MSAL token-cache
        file-lock contention.
        """
        try:
            return _get_or_create_credential()
        except Exception as e:
            logger.error("Failed to get credential: %s", e)
            return None

    async def _ensure_folder_exists(self, folder_path: str) -> bool:
        """
        Ensure folder path exists, creating nested folders as needed.
        folder_path: e.g., "pelvic-2d-points-backup/username/2024-01-15"
        """
        if not self._client:
            return False

        parts = folder_path.strip("/").split("/")
        current_path = ""

        for part in parts:
            parent_path = current_path if current_path else "root"
            current_path = f"{current_path}/{part}" if current_path else part

            try:
                # Check if folder exists by trying to get it
                item_path = f"root:/{current_path}:" if current_path else "root"
                await (
                    self._client.drives.by_drive_id(SHAREPOINT_DRIVE_ID)
                    .items.by_drive_item_id(item_path)
                    .get()
                )
            except Exception:
                # Folder doesn't exist, create it
                try:
                    request_body = DriveItem(
                        name=part,
                        folder=Folder(),
                        additional_data={
                            "@microsoft.graph" + ".conflictBehavior": "fail"
                        },
                    )

                    parent_item_id = (
                        f"root:/{'/'.join(current_path.split('/')[:-1])}:"
                        if "/" in current_path
                        else "root"
                    )

                    await (
                        self._client.drives.by_drive_id(SHAREPOINT_DRIVE_ID)
                        .items.by_drive_item_id(parent_item_id)
                        .children.post(request_body)
                    )
                    logger.info(f"Created folder: {current_path}")
                except Exception as e:
                    # Only ignore the benign "already exists" race condition;
                    # re-raise everything else so callers see the real error.
                    if "nameAlreadyExists" in str(e):
                        logger.debug("Folder already exists: %s", current_path)
                    else:
                        raise

        return True

    async def _upload_file_async(self, local_path: Path, remote_folder: str) -> bool:
        """Upload a file to the specified remote folder."""
        if not self._client:
            return False

        try:
            with open(local_path, "rb") as f:
                file_content = f.read()

            dest_file_name = local_path.name
            drive_item_path = f"root:/{remote_folder}/{dest_file_name}:"

            uploaded_file: Optional[DriveItem] = await (
                self._client.drives.by_drive_id(SHAREPOINT_DRIVE_ID)
                .items.by_drive_item_id(drive_item_path)
                .content.put(file_content)
            )

            if uploaded_file and uploaded_file.name:
                logger.info(f"Uploaded: {uploaded_file.name} to {remote_folder}")
                return True
            return False

        except Exception as e:
            logger.error(f"Failed to upload {local_path}: {e}")
            return False

    async def _backup_file_async(self, local_path: Path) -> bool:
        """
        Backup a file to OneDrive with user/date folder structure.

        Uploads to: pelvic-2d-points-backup/<username>/<YYYY-MM-DD>/<filename>
        """
        # Note: _ensure_initialized must be called BEFORE entering async context
        # (it's called in upload_backup_sync before run_until_complete)
        if not self._client:
            logger.error("OneDrive client not initialized")
            return False

        username = get_safe_username()
        date_folder = get_date_folder()
        remote_folder = f"{BASE_BACKUP_FOLDER}/{username}/{date_folder}"

        # Try to upload directly - OneDrive API creates parent folders automatically
        # when using path-based addressing like "root:/folder/subfolder/file.txt:"
        return await self._upload_file_async(local_path, remote_folder)

    def upload_backup(
        self, file_path: str | Path, callback: Optional[Callable[[bool], None]] = None
    ) -> None:
        """
        Upload a file to OneDrive backup in background thread.

        Args:
            file_path: Path to file to upload (db or csv)
            callback: Optional callback(success: bool) when complete

        Non-blocking - runs in background thread.
        """
        path = Path(file_path)
        if not path.exists():
            logger.warning(f"File does not exist: {path}")
            if callback:
                callback(False)
            return

        def _run():
            # Use sync upload which creates fresh client per thread
            success = self.upload_backup_sync(path, timeout=30.0)
            if callback:
                callback(success)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    def _create_fresh_client(self) -> Optional[GraphServiceClient]:
        """
        Create a fresh Graph client for the current thread.

        The MS Graph SDK uses httpx which has thread/event-loop affinity,
        so we need a fresh GraphServiceClient for each thread. However,
        the DeviceCodeCredential is reused — creating many credential
        instances causes MSAL token-cache file-lock contention and hangs.
        """
        try:
            logger.info("Creating fresh client for this thread...")

            credential = self._get_shared_credential()
            if credential is None:
                return None

            client = GraphServiceClient(credentials=credential, scopes=SCOPES)
            logger.info("Fresh client created successfully")
            return client

        except Exception as e:
            logger.error("Failed to create fresh client: %s", e)
            return None

    # Maximum number of upload attempts (including the first).
    _UPLOAD_MAX_RETRIES = 3
    # Delays between retries (index 0 = after first failure, etc.).
    _UPLOAD_RETRY_DELAYS = (1.0, 2.0, 4.0)

    @staticmethod
    def _is_transient_error(exc: Exception) -> bool:
        """Return True if *exc* represents a transient / retryable failure.

        Transient errors include network timeouts, connection resets, and
        HTTP 429 / 5xx responses.  Client errors (4xx except 429) are
        considered permanent.
        """
        exc_str = str(exc).lower()
        # Network-level errors
        if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
            return True
        # asyncio.TimeoutError is a subclass of TimeoutError, already covered
        # HTTP status hints in exception messages from the Graph SDK
        for code in ("429", "500", "502", "503", "504"):
            if code in exc_str:
                return True
        return False

    def upload_backup_sync(self, file_path: str | Path, timeout: float = 30.0) -> bool:
        """
        Upload a file to OneDrive backup synchronously with timeout.

        Use this for critical saves (e.g., on window close) where
        you need to ensure the upload completes before continuing.

        Transient failures (timeouts, network errors, HTTP 429/5xx) are
        retried up to ``_UPLOAD_MAX_RETRIES`` times with exponential
        backoff.  Permanent client errors (4xx except 429) fail immediately.

        Args:
            file_path: Path to file to upload
            timeout: Maximum seconds to wait for each upload attempt

        Returns True on success, False on failure or timeout.
        """
        path = Path(file_path)
        if not path.exists():
            logger.warning("File does not exist: %s", path)
            return False

        logger.info("Starting sync upload: %s", path.name)

        last_exc: Optional[Exception] = None

        for attempt in range(self._UPLOAD_MAX_RETRIES):
            if attempt > 0:
                delay = self._UPLOAD_RETRY_DELAYS[attempt - 1]
                logger.info(
                    "Retry %d/%d for %s after %.1fs backoff...",
                    attempt + 1,
                    self._UPLOAD_MAX_RETRIES,
                    path.name,
                    delay,
                )
                time.sleep(delay)

            try:
                client = self._create_fresh_client()
                if not client:
                    logger.warning("No client available, skipping upload")
                    return False

                loop = asyncio.SelectorEventLoop()

                async def do_upload():
                    logger.info("Reading file: %s", path.name)
                    with open(path, "rb") as f:
                        file_content = f.read()
                    logger.info("File size: %d bytes", len(file_content))

                    username = get_safe_username()
                    date_folder = get_date_folder()
                    remote_folder = f"{BASE_BACKUP_FOLDER}/{username}/{date_folder}"
                    drive_item_path = f"root:/{remote_folder}/{path.name}:"

                    logger.info("Uploading to: %s", drive_item_path)

                    uploaded_file = await (
                        client.drives.by_drive_id(SHAREPOINT_DRIVE_ID)
                        .items.by_drive_item_id(drive_item_path)
                        .content.put(file_content)
                    )

                    logger.info(
                        "Upload complete: %s",
                        uploaded_file.name if uploaded_file else "unknown",
                    )
                    return uploaded_file is not None

                logger.info("Running upload with %.1fs timeout...", timeout)
                coro = asyncio.wait_for(do_upload(), timeout=timeout)
                try:
                    success = loop.run_until_complete(coro)
                finally:
                    loop.close()
                logger.info(
                    "Sync upload result: %s", "success" if success else "failed"
                )
                return success

            except asyncio.TimeoutError as e:
                last_exc = e
                logger.warning(
                    "Upload attempt %d timed out after %.1fs for %s",
                    attempt + 1, timeout, path,
                )
                # Timeout is transient — retry if we have attempts left
                continue
            except Exception as e:
                last_exc = e
                if self._is_transient_error(e):
                    logger.warning(
                        "Transient error on upload attempt %d for %s: %s",
                        attempt + 1, path, e,
                    )
                    continue
                else:
                    # Permanent error — no point retrying
                    logger.error("Permanent sync backup error: %s", e, exc_info=True)
                    return False

        # All retries exhausted
        logger.error(
            "Upload failed after %d attempts for %s: %s",
            self._UPLOAD_MAX_RETRIES, path, last_exc,
        )
        return False

    def backup_multiple(
        self,
        file_paths: Sequence[str | Path],
        callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """
        Backup multiple files in background.

        Args:
            file_paths: List of file paths to backup
            callback: Optional callback(success_count, total_count) when complete
        """

        def _run():
            success_count = 0
            total = len(file_paths)

            for path in file_paths:
                p = Path(path)
                if p.exists():
                    # Use sync upload which creates fresh client per thread
                    if self.upload_backup_sync(p, timeout=30.0):
                        success_count += 1

            if callback:
                callback(success_count, total)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()


# Convenience function for simple usage
_backup_instance: Optional[OneDriveBackup] = None


def get_backup_instance() -> OneDriveBackup:
    """Get or create the singleton backup instance."""
    global _backup_instance
    if _backup_instance is None:
        _backup_instance = OneDriveBackup()
    return _backup_instance


def backup_to_onedrive(
    file_path: str | Path, callback: Optional[Callable[[bool], None]] = None
) -> None:
    """
    Convenience function to backup a file to OneDrive.

    Usage:
        from auth import backup_to_onedrive
        backup_to_onedrive("/path/to/file.db")
    """
    get_backup_instance().upload_backup(file_path, callback)
