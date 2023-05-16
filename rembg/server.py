from enum import Enum
from typing import Optional, Tuple, cast, Annotated

import aiohttp
import uvicorn
from asyncer import asyncify
from fastapi import Depends, FastAPI, File, Form, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response

from . import _version
from .bg import remove, clothes_seg_to_firebase, ClothesType
from .session_base import BaseSession
from .session_factory import new_session

sessions: dict[str, BaseSession] = {}
tags_metadata = [
    {
        "name": "Background Removal",
        "description": "Endpoints that perform background removal with different image sources.",
        "externalDocs": {
            "description": "GitHub Source",
            "url": "https://github.com/danielgatis/rembg",
        },
    },
]
app = FastAPI(
    title="Rembg",
    description="Rembg is a tool to remove images background. That is it.",
    version=_version.get_versions()["version"],
    contact={
        "name": "Daniel Gatis",
        "url": "https://github.com/danielgatis",
        "email": "danielgatis@gmail.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://github.com/danielgatis/rembg/blob/main/LICENSE.txt",
    },
    openapi_tags=tags_metadata,
)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def start_server(port: int, log_level: str, threads: int) -> None:
    uvicorn.run("rembg.server:app", host="0.0.0.0", port=port, workers=threads, log_level=log_level)


class ModelType(str, Enum):
    u2net = "u2net"
    u2netp = "u2netp"
    u2net_human_seg = "u2net_human_seg"
    u2net_cloth_seg = "u2net_cloth_seg"
    silueta = "silueta"
    isnet_general_use = "isnet-general-use"


class CommonQueryParams:
    def __init__(
        self,
        model: ModelType = Query(
            default=ModelType.u2net,
            description="Model to use when processing image",
        ),
        a: bool = Query(default=False, description="Enable Alpha Matting"),
        af: int = Query(
            default=240,
            ge=0,
            le=255,
            description="Alpha Matting (Foreground Threshold)",
        ),
        ab: int = Query(
            default=10,
            ge=0,
            le=255,
            description="Alpha Matting (Background Threshold)",
        ),
        ae: int = Query(
            default=10, ge=0, description="Alpha Matting (Erode Structure Size)"
        ),
        om: bool = Query(default=False, description="Only Mask"),
        ppm: bool = Query(default=False, description="Post Process Mask"),
        bgc: Optional[str] = Query(default=None, description="Background Color"),
    ):
        self.model = model
        self.a = a
        self.af = af
        self.ab = ab
        self.ae = ae
        self.om = om
        self.ppm = ppm
        self.bgc = (
            cast(Tuple[int, int, int, int], tuple(map(int, bgc.split(","))))
            if bgc
            else None
        )


class CommonQueryPostParams:
    def __init__(
        self,
        model: ModelType = Form(
            default=ModelType.u2net,
            description="Model to use when processing image",
        ),
        a: bool = Form(default=False, description="Enable Alpha Matting"),
        af: int = Form(
            default=240,
            ge=0,
            le=255,
            description="Alpha Matting (Foreground Threshold)",
        ),
        ab: int = Form(
            default=10,
            ge=0,
            le=255,
            description="Alpha Matting (Background Threshold)",
        ),
        ae: int = Form(
            default=10, ge=0, description="Alpha Matting (Erode Structure Size)"
        ),
        om: bool = Form(default=False, description="Only Mask"),
        ppm: bool = Form(default=False, description="Post Process Mask"),
        bgc: Optional[str] = Query(default=None, description="Background Color"),
    ):
        self.model = model
        self.a = a
        self.af = af
        self.ab = ab
        self.ae = ae
        self.om = om
        self.ppm = ppm
        self.bgc = (
            cast(Tuple[int, int, int, int], tuple(map(int, bgc.split(","))))
            if bgc
            else None
        )


def im_without_bg(content: bytes, commons: CommonQueryParams) -> Response:
    return Response(
        remove(
            content,
            session=sessions.setdefault(
                commons.model.value, new_session(commons.model.value)
            ),
            alpha_matting=commons.a,
            alpha_matting_foreground_threshold=commons.af,
            alpha_matting_background_threshold=commons.ab,
            alpha_matting_erode_size=commons.ae,
            only_mask=commons.om,
            post_process_mask=commons.ppm,
            bgcolor=commons.bgc,
        ),
        media_type="image/png",
    )


@app.get(
    path="/",
    tags=["Background Removal"],
    summary="Remove from URL",
    description="Removes the background from an image obtained by retrieving an URL.",
)
async def get_index(
    url: str = Query(
        default=..., description="URL of the image that has to be processed."
    ),
    commons: CommonQueryParams = Depends(),
):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            file = await response.read()
            return await asyncify(im_without_bg)(file, commons)


@app.post(
    path="/",
    tags=["Background Removal"],
    summary="Remove from Stream",
    description="Removes the background from an image sent within the request itself.",
)
async def post_index(
    file: bytes = File(
        default=...,
        description="Image file (byte stream) that has to be processed.",
    ),
    commons: CommonQueryPostParams = Depends(),
):
    return await asyncify(im_without_bg)(file, commons)  # type: ignore


@app.post(
    path="/seg_clothes",
    tags=["segregation", "clothes"],
    summary="Remove from filestream.",
    description="Remove Everything but clothes.",
)
async def post_seg_clothes(
    file: Annotated[UploadFile, File(description="Image file (byte stream) that has to be processed.")],
    uid: Annotated[str, Form()],
    smooth_edges: Annotated[bool, Form()] = False,
    include: Annotated[list[ClothesType] | None, Query(description="Types of clothes to include")] = None,
):
    return clothes_seg_to_firebase(await file.read(), uid=uid, included=include, post_process_mask=smooth_edges)
